import os
import sys
from datetime import datetime
from typing import List, Union
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np
import pandas as pd
from scipy import signal
import seaborn as sns
from pathlib import Path
import holidays
import pickle
from sklearn.decomposition import PCA
import re
import glob
import warnings
from pandas.errors import SettingWithCopyWarning
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data.encoders import GroupNormalizer, NaNLabelEncoder
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from transformers import BertTokenizer, BertModel
import pickle
from pytorch_lightning import LightningModule
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from loguru import logger
from pytorch_forecasting.metrics.point import CrossEntropy, PoissonLoss, TweedieLoss
import argparse
from clip import clip
from PIL import Image
from train_util import read_train_df, split_by_product
warnings.simplefilter("error", category=SettingWithCopyWarning)

config = {
    "model": "tft",
    "dataset_path": "mys.csv",
    "dataset_image_path": "mys-images",
    "train_logger": TensorBoardLogger(save_dir='logs', name='train'),

    "max_prediction_length": 2,
    "max_encoder_length": 30,
    "train_ratio": 0.8,
    "time_varying_unknown_reals": ['sale', "log_sale", 'views', 'avg_sale_by_id'],
    "dynamic_features": [
            'product_time_idx', 'weekday_cos', 'weekday_sin', 'week_cos', 'week_sin',
            'weekend', 'holidays', 'month', 'hour_of_day', 'sale', 'log_sale', 'price',
            'duration', 'views', 'avg_sale_by_id'
        ],

    "batch_size": 64,
    "train_batch_size": 10,
    "gradient_clip_val": 0.1,
    "learning_rate": 0.001,
    "hidden_size": 128,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 32,
    "share_single_variable_networks": False,
    "causal_attention": False,
    "use_pca": False
}

logger.info("loading model....")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
logger.info(f"loaded model: bert, clip.")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def get_clip_text_embedding(text):
    text_inputs = clip.tokenize([text]).to(device)
    clip_model.eval()
    with torch.no_grad():
        outputs = clip_model.encode_text(text_inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)

    return outputs.cpu().numpy().flatten()

def get_clip_image_embedding(image_name):
    image_path = os.path.join(config["dataset_image_path"], image_name)
    image = Image.open(image_path)
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    clip_model.eval()
    with torch.no_grad():
        outputs = clip_model.encode_image(image_input)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)

    return outputs.cpu().numpy().flatten()

def get_pca_feature(df, feature_name, dimension_num):
    embedding_matrix = np.vstack(df[feature_name].values)

    pca = PCA(n_components=dimension_num)
    reduced_embeddings = pca.fit_transform(embedding_matrix)

    for i in range(reduced_embeddings.shape[1]):
        df[f'{feature_name}_pca_{i+1}'] = reduced_embeddings[:, i]
    return [f"{feature_name}_pca_{i+1}" for i in range(reduced_embeddings.shape[1])]

def process_train_data(config):
    train = read_train_df(config)

    logger.info("process text and image feature...")
    if config["model"] == "tft":
        pass
    elif config["model"] == "btft":
        train["bert_embedding"] = train["title"].apply(get_bert_embedding)
        bert_embedding_size = train["bert_embedding"].iloc[0].shape[0]
        bert_embedding_df = pd.DataFrame(train["bert_embedding"].tolist(), index=train.index)
        train = pd.concat([train, bert_embedding_df.add_prefix('bert_embedding_')], axis=1)

        new_features = [f"bert_embedding_{i}" for i in range(len(bert_embedding_df.columns))]
        if config["use_pca"]:
            new_features = get_pca_feature(train, "bert_embedding", 10)
        config["time_varying_unknown_reals"] = ['sale',"log_sale" ,'price', 'duration', 'views', 'avg_sale_by_id'] + new_features
    elif config["model"] == "mtft":
        train["clip_text_embedding"] = train["title"].apply(get_clip_text_embedding)
        clip_text_embedding_size = train["clip_text_embedding"].iloc[0].shape[0]
        clip_text_embedding_df = pd.DataFrame(train["clip_text_embedding"].tolist(), index=train.index)
        train = pd.concat([train, clip_text_embedding_df.add_prefix('clip_text_embedding_')], axis=1)

        train["clip_image_embedding"] = train["live_screenshot"].apply(get_clip_image_embedding)
        clip_image_embedding_size = train["clip_image_embedding"].iloc[0].shape[0]
        clip_image_embedding_df = pd.DataFrame(train["clip_image_embedding"].tolist(), index=train.index)
        train = pd.concat([train, clip_image_embedding_df.add_prefix('clip_image_embedding_')], axis=1)

        new_features = [f"clip_text_embedding_{i}" for i in range(len(clip_text_embedding_df.columns))] + [f"clip_image_embedding_{i}" for i in range(len(clip_image_embedding_df.columns))]
        if config["use_pca"]:
            new_features = get_pca_feature(train, "clip_text_embedding", 5)
            new_features += get_pca_feature(train, "clip_image_embedding", 5)
        config["time_varying_unknown_reals"] = ['sale',"log_sale" ,'price', 'duration', 'views', 'avg_sale_by_id'] + new_features
    else:
        raise(f"Unkown model: {config['model']}!")
    logger.info("process text and image feature done.")
    logger.info(f"time_varying_unknown_reals length: {len(config['time_varying_unknown_reals'])}")

    return train

def train(config, training, validation):

    training_dataset = TimeSeriesDataSet(
        training,
        time_idx="product_time_idx",
        target="sale",
        group_ids=["id"],
        min_encoder_length=config["max_encoder_length"] // 2,
        max_encoder_length=config["max_encoder_length"],
        min_prediction_length=1,
        max_prediction_length=config["max_prediction_length"],
        static_categoricals=["id"],
        static_reals=[],
        time_varying_known_categoricals=['hour_of_day', 'month'],
        time_varying_known_reals=['price', 'duration', 'weekday_cos', 'weekday_sin', 'week_cos', 'week_sin', 'weekend', 'holidays'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=config["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
        categorical_encoders={"month": NaNLabelEncoder(add_nan=True), "hour_of_day": NaNLabelEncoder(add_nan=True)},
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation, predict=True, stop_randomization=True)

    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=4)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=config["batch_size"] * 10, num_workers=4)

    lr_logger = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=config["gradient_clip_val"],
        limit_train_batches=30,
        log_every_n_steps=10,
        callbacks=[lr_logger, early_stop_callback],
        logger=config["train_logger"],
        precision=32
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["attention_head_size"],
        dropout=config["dropout"],
        hidden_continuous_size=config["hidden_continuous_size"],
        output_size=7,  # Number of quantiles
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        share_single_variable_networks=config["share_single_variable_networks"],
        causal_attention=config["causal_attention"],
        optimizer='adam'
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    if isinstance(predictions.y, tuple):
        print("predictions.y is a tuple.")
        # Assuming the true values are the first element of the tuple
        true_values = predictions.y[0]
    elif isinstance(predictions.y, torch.Tensor):
        print("predictions.y is a Tensor.")
        true_values = predictions.y
    else:
        print(f"predictions.y is of an unknown type: {type(predictions.y)}")

    mse_value = torch.mean((predictions.output - true_values) ** 2)
    mae_value = torch.mean(torch.abs(predictions.output - true_values))
    rmse_value = torch.sqrt(mse_value)

    logger.info(f"mae_value: {mae_value}, mse_value: {mse_value}, rmse_value: {rmse_value}")
    return mae_value, mse_value, rmse_value

def train_with_optuna():
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=50,
        max_epochs=50,
        gradient_clip_val_range=(0.05, 0.5),
        hidden_size_range=(16, 64),
        hidden_continuous_size_range=(2, 32),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.05),
        dropout_range=(0.05, 0.5),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )

    logger.info(study.best_trial.params)
    train(study.best_trial.params)

def train_batch(_config):
    data = []
    config.update(_config)
    mae_total = []
    mse_total = []
    rmse_total = []

    dataset = process_train_data(config)
    training, validation = split_by_product(config, dataset)
    for i in range(config["train_batch_size"]):
        mae, mse, rmse = train(config, training, validation)
        mae_total += [float(mae)]
        mse_total += [float(mse)]
        rmse_total += [float(rmse)]
        data.append(f"Train {i+1}. mae_value: {mae}, mse_value: {mse}, rmse_value: {rmse}")
    for e in data:
        logger.info(e)
    logger.info(_config)
    logger.info(f"Train {config['train_batch_size']} times. avg_mae: {sum(mae_total) / float(config['train_batch_size'])}±{np.std(np.array(mae_total))}, avg_mse: {sum(mse_total) / float(config['train_batch_size'])}±{np.std(np.array(mse_total))}, avg_rmse: {sum(rmse_total) / float(config['train_batch_size'])}±{np.std(np.array(rmse_total))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tft")
    parser.add_argument("--model", type=str, default="tft", help="model")
    parser.add_argument("--dataset_path", type=str, default="US_8903_Picture_cleaned.csv", help="dataset_path")
    parser.add_argument("--dataset_image_path", type=str, default="US_8903_Picture", help="dataset_image_path")
    parser.add_argument("--max_encoder_length", type=int, default=30, help="max_encoder_length")
    parser.add_argument("--max_prediction_length", type=int, default=2, help="max_prediction_length")
    parser.add_argument("--share_single_variable_networks", type=lambda x: (str(x).lower() == 'true'), default=False, help="share_single_variable_networks")
    parser.add_argument("--causal_attention", type=lambda x: (str(x).lower() == 'true'), default=False, help="causal_attention")
    parser.add_argument("--use_pca", type=lambda x: (str(x).lower() == 'true'), default=False, help="use_pca")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning_rate")
    parser.add_argument("--train_batch_size", type=int, default=10, help="train_batch_size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train_ratio")

    args = parser.parse_args()
    train_batch({
        "model": args.model,
        "dataset_path": args.dataset_path,
        "dataset_image_path": args.dataset_image_path,
        "max_encoder_length": args.max_encoder_length,
        "max_prediction_length": args.max_prediction_length,
        "share_single_variable_networks": args.share_single_variable_networks,
        "causal_attention": args.causal_attention,
        "use_pca": args.use_pca,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "train_ratio": args.train_ratio,
        })