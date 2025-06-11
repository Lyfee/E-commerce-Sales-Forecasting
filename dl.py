import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torchmetrics
import holidays
from train_util import read_data_and_split, EarlyStopping, MetricsCalculator
from loguru import logger
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SalesCNN(nn.Module):
    def __init__(self, input_size, output_size, num_ids):
        super(SalesCNN, self).__init__()

        self.id_embedding = nn.Embedding(num_ids, 4)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size+4, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten_size = None

        self.fc_layers = nn.Sequential(
            nn.Linear(1, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size)
        )

    def _calculate_flatten_size(self, input_shape):
        dummy_input = torch.randn(1, input_shape[1], input_shape[0]).to(self.conv_layers[0].weight.device) # Move dummy input to the same device as weights
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.view(1, -1).size(-1)

    def forward(self, x, ids):
        batch_size, seq_len, _ = x.size()
        embedded_ids = self.id_embedding(ids)
        embedded_ids = embedded_ids.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([x, embedded_ids], dim=2)

        combined = combined.permute(0, 2, 1)

        conv_out = self.conv_layers(combined)

        flattened = conv_out.view(batch_size, -1)

        if self.flatten_size is None or self.fc_layers[0].in_features != flattened.shape[1]:
            self.flatten_size = flattened.shape[1]
            self.fc_layers[0] = nn.Linear(self.flatten_size, 1024).to(self.fc_layers[0].weight.device)

        return self.fc_layers(flattened)

class SalesLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_ids, hidden_size=128, num_layers=2):
        super(SalesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.id_embedding = nn.Embedding(num_ids, 4)

        self.lstm = nn.LSTM(
            input_size=input_size + 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5 if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, ids):
        embedded_ids = self.id_embedding(ids)
        embedded_ids = embedded_ids.unsqueeze(1)
        repeated_embeddings = embedded_ids.repeat(1, x.size(1), 1)

        combined = torch.cat([x, repeated_embeddings], dim=2)
        out, _ = self.lstm(combined)
        out = self.fc(out[:, -1, :])
        return out

class SalesGRU(nn.Module):
    def __init__(self, input_size, output_size, num_ids, hidden_size=128, num_layers=2):
        super(SalesGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.id_embedding = nn.Embedding(num_ids, 4)

        self.gru = nn.GRU(
            input_size=input_size + 4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5 if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, ids):
        embedded_ids = self.id_embedding(ids)
        embedded_ids = embedded_ids.unsqueeze(1)
        repeated_embeddings = embedded_ids.repeat(1, x.size(1), 1)

        combined = torch.cat([x, repeated_embeddings], dim=2)
        out, _ = self.gru(combined)
        out = self.fc(out[:, -1, :])
        return out

config = {
    "dataset_path": "US_8903_Picture_cleaned.csv",
    'max_prediction_length': 3,
    'batch_size': 64,
    'learning_rate': 0.005,
    'epochs': 2000,
    "model": "cnn",
    "num_workers": 4,
    "train_ratio": 0.8,
    "max_encoder_length": 30,
    "train_batch_size": 10,
    "time_varying_unknown_reals": ['sale',"log_sale" ,'price', 'duration', 'views', 'avg_sale_by_id'],

    "dynamic_features": [
            'product_time_idx', 'weekday_cos', 'weekday_sin', 'week_cos', 'week_sin',
            'weekend', 'holidays', 'month', 'hour_of_day', 'sale', 'log_sale', 'price',
            'duration', 'views', 'avg_sale_by_id'
        ]
}

def train(config):
    train_loader, val_loader, ids = read_data_and_split(config)

    if config["model"] == "cnn":
        model = SalesCNN(
            input_size=len(config["dynamic_features"]),
            output_size=config['max_prediction_length'],
            num_ids=len(ids)
        ).to(device)
    elif config["model"] == "lstm":
        model = SalesLSTM(
            input_size=len(config["dynamic_features"]),
            output_size=config['max_prediction_length'],
            num_ids=len(ids)
        ).to(device)
    elif config["model"] == "gru":
        model = SalesGRU(
            input_size=len(config["dynamic_features"]),
            output_size=config['max_prediction_length'],
            num_ids=len(ids)
        ).to(device)
    else:
        raise(f"Unkown model: {config['model']}!")

    criterion = nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.1)

    early_stopping = EarlyStopping(patience=20, delta=0.01)

    best_avg_val_loss = 1000000.0
    best_epoch_val_metrics = None
    for epoch in range(config['epochs']):
        model.train()
        train_metrics = MetricsCalculator()
        train_loss = 0

        for X_batch, y_batch, id_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            id_batch = id_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, id_batch)  # [batch_size, pred_len]

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metrics.update(y_batch, outputs)

        epoch_train_metrics = train_metrics.compute()
        train_metrics.reset()

        model.eval()
        val_metrics = MetricsCalculator()
        val_loss = 0

        with torch.no_grad():
            for X_val, y_val, id_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                id_val = id_val.to(device)

                outputs = model(X_val, id_val)
                loss = criterion(outputs, y_val)

                val_loss += loss.item()
                val_metrics.update(y_val, outputs)

        epoch_val_metrics = val_metrics.compute()

        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"MAE: {epoch_train_metrics['mae']:.4f} | "
              f"MSE: {epoch_train_metrics['mse']:.4f} | "
              f"RMSE: {epoch_train_metrics['rmse']:.4f}")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"MAE: {epoch_val_metrics['mae']:.4f} | "
              f"MSE: {epoch_val_metrics['mse']:.4f} | "
              f"RMSE: {epoch_val_metrics['rmse']:.4f}\n")

        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            best_epoch_val_metrics = epoch_val_metrics

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            break

    return best_epoch_val_metrics['mae'], best_epoch_val_metrics['mse'], best_epoch_val_metrics['rmse']

def train_batch(_config):
    data = []
    config.update(_config)
    mae_total = []
    mse_total = []
    rmse_total = []

    for i in range(config["train_batch_size"]):
        mae, mse, rmse = train(config)
        mae_total += [float(mae)]
        mse_total += [float(mse)]
        rmse_total += [float(rmse)]
        data.append(f"Train {i+1}. mae_value: {mae}, mse_value: {mse}, rmse_value: {rmse}")
    for e in data:
        logger.info(e)
    logger.info(_config)
    logger.info(f"Train {config['train_batch_size']} times. avg_mae: {sum(mae_total) / float(config['train_batch_size'])}±{np.std(np.array(mae_total))}, avg_mse: {sum(mse_total) / float(config['train_batch_size'])}±{np.std(np.array(mse_total))}, avg_rmse: {sum(rmse_total) / float(config['train_batch_size'])}±{np.std(np.array(rmse_total))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="traditional_model")
    parser.add_argument("--model", type=str, default="cnn", help="model")
    parser.add_argument("--dataset_path", type=str, default="US_8903_Picture_cleaned.csv", help="dataset_path")
    parser.add_argument("--max_encoder_length", type=int, default=30, help="max_encoder_length")
    parser.add_argument("--max_prediction_length", type=int, default=2, help="max_prediction_length")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning_rate")
    parser.add_argument("--train_batch_size", type=int, default=10, help="train_batch_size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train_ratio")

    args = parser.parse_args()
    train_batch({
        "model": args.model,
        "dataset_path": args.dataset_path,
        "max_encoder_length": args.max_encoder_length,
        "max_prediction_length": args.max_prediction_length,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "train_ratio": args.train_ratio,
        })