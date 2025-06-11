import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
import numpy as np
from loguru import logger
import torchmetrics
import holidays
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class TraditionalModelTimeSeriesDataset(Dataset):
    def __init__(self, data, config,
                 scalers=None, mode='train'):
        self.samples = []
        self.max_encoder_length = config["max_encoder_length"]
        self.min_encoder_length = self.max_encoder_length // 5
        self.max_prediction_length = config["max_prediction_length"]
        self.id_to_index = {id: i for i, id in enumerate(data['id'].unique())}
        self.mode = mode

        self.dynamic_features = config["dynamic_features"]
        self.static_features = ['id']
        self.target = 'sale'

        self.grouped_data = data.groupby('id')

        self.scalers = scalers if scalers is not None else {}

        if mode == 'train':
            self._create_scalers()

        elif mode == 'valid' and not self.scalers:
            raise ValueError("must set scalers if mode == valid!")

        self._generate_samples()

    def _create_scalers(self):
        for product_id, product_data in self.grouped_data:
            scaler = StandardScaler()
            scaler.fit(product_data[self.dynamic_features])

            self.scalers[product_id] = scaler

    def _generate_samples(self):
        for product_id, product_data in self.grouped_data:
            scaler = self.scalers[product_id]
            scaled_data = scaler.transform(product_data[self.dynamic_features])

            num_rows = len(product_data)

            if self.mode == 'train':
                for i in range(num_rows - self.max_encoder_length - self.max_prediction_length + 1):
                    input_seq = scaled_data[i:i+self.max_encoder_length]
                    target = product_data[self.target].values[
                        i+self.max_encoder_length : i+self.max_encoder_length+self.max_prediction_length
                    ]
                    self.samples.append({
                        'X': input_seq,
                        'y': target,
                        'id': product_id
                    })

            elif self.mode == 'valid':
                encoder_length = self.max_encoder_length
                i = num_rows - encoder_length - self.max_prediction_length
                input_seq = scaled_data[i:i+encoder_length]
                target = product_data[self.target].values[
                    i+encoder_length : i+encoder_length+self.max_prediction_length
                ]

                if len(input_seq) == encoder_length:
                    self.samples.append({
                        'X': input_seq,
                        'y': target,
                        'id': product_id
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['X']),
            torch.FloatTensor(sample['y']),
            torch.LongTensor([self.id_to_index[sample['id']]])
        )

def collate_fn(batch):
    X_batch = torch.stack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    id_batch = torch.cat([item[2] for item in batch])
    return X_batch, y_batch, id_batch

def read_train_df(config):
    train_df = pd.read_csv(config["dataset_path"])

    train_df.loc[:, 'id'] = train_df['id'].astype(str)
    train_df.loc[:, 'title'] = train_df['title'].astype(str)
    train_df = train_df.reset_index(drop=True)
    train_df['create_time'] = pd.to_datetime(train_df['create_time'])

    train_df["month"] = train_df.create_time.dt.month.astype(str).astype("category")
    train_df["log_sale"] = np.log(train_df.sale + 1e-8)
    train_df["avg_sale_by_id"] = train_df.groupby(["id"], observed=True).sale.transform("mean")
    train_df['hour_of_day'] = train_df['create_time'].dt.hour.astype(str).astype("category")

    # remove short length product
    group_sizes = train_df.groupby('id').size()
    groups_greater_num = (config["max_encoder_length"] + config["max_prediction_length"]) // (1 - config["train_ratio"])
    groups_greater = group_sizes[group_sizes > groups_greater_num]
    number_of_groups = len(groups_greater)
    train_df = train_df[train_df['id'].isin(groups_greater.index.tolist())]

    return train_df

def split_by_product(config, df):
    train_dfs = []
    val_dfs = []

    product_ids = df['id'].unique()

    df = df.sort_values(by=['id', 'create_time'], ascending=[True, True]).reset_index(drop=True)
    df['product_time_idx'] = df.index + 1

    for product_id in product_ids:
        product_data = df[df['id'] == product_id].copy()
        train_size = int(len(product_data) * config["train_ratio"])

        train_df = product_data.iloc[:train_size].copy()
        val_df = product_data.iloc[train_size:].copy()

        train_dfs.append(train_df)
        val_dfs.append(val_df)

    training = pd.concat(train_dfs).reset_index(drop=True)
    validation = pd.concat(val_dfs).reset_index(drop=True)

    return training, validation

def read_data_and_split(config):

    dataset = read_train_df(config)
    training, validation = split_by_product(config, dataset)
    training_dataset = TraditionalModelTimeSeriesDataset(
        training,
        config=config,
        mode='train'
    )

    validation_dataset = TraditionalModelTimeSeriesDataset(
        validation,
        config=config,
        scalers=training_dataset.scalers,
        mode='valid'
    )

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, dataset["id"].unique()

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001, verbose=True, warmup=0):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.epoch = 0

    def __call__(self, val_loss, model):
        self.epoch += 1

        if self.epoch <= self.warmup:
            if self.verbose:
                print(f'Warmup phase: epoch {self.epoch}/{self.warmup} - skipping early stop check')
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pth')
        self.val_loss_min = val_loss

class MetricsCalculator:
    def __init__(self):
        self.mae = torchmetrics.MeanAbsoluteError().to(device)
        self.mse = torchmetrics.MeanSquaredError().to(device)

    def update(self, y_true, y_pred):
        self.mae.update(y_pred.view(-1), y_true.view(-1))
        self.mse.update(y_pred.view(-1), y_true.view(-1))

    def compute(self):
        return {
            "mae": self.mae.compute().item(),
            "mse": self.mse.compute().item(),
            "rmse": torch.sqrt(self.mse.compute()).item()
        }

    def reset(self):
        self.mae.reset()
        self.mse.reset()