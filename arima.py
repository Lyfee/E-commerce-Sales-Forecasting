import pandas as pd
import numpy as np
from pmdarima import auto_arima, ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from train_util import read_train_df
import argparse
import warnings
warnings.simplefilter('ignore')

# config
config = {
    'max_prediction_length': 12,
    'seasonal_period': 7,
    'model': "arima",
    "train_ratio": 0.8,
    'arima_order': (1,1,0),
    'sarima_seasonal_order': (1,1,0,12),
    'dynamic_features': [
            'weekday_cos', 'weekday_sin', 'week_cos', 'week_sin',
            'weekend', 'holidays', 'month', 'hour_of_day', 'sale', 'log_sale', 'price',
            'duration', 'views', 'avg_sale_by_id'
        ],
    'target': 'sale',
    'numeric_features': [
            'weekday_cos', 'weekday_sin', 'week_cos', 'week_sin',
            'weekend', 'holidays', 'month', 'hour_of_day', 'sale', 'log_sale', 'price',
            'duration', 'views', 'avg_sale_by_id'
        ],
    "country": "mys"
}

def preprocess_data(df):
    def add_time_features(df_inner):
        df_inner['weekday_cos'] = np.cos(2 * np.pi * df_inner['create_time'].dt.dayofweek / 7)
        df_inner['weekday_sin'] = np.sin(2 * np.pi * df_inner['create_time'].dt.dayofweek / 7)
        df_inner['week_cos'] = np.cos(2 * np.pi * df_inner['create_time'].dt.isocalendar().week / 52)
        df_inner['week_sin'] = np.sin(2 * np.pi * df_inner['create_time'].dt.isocalendar().week / 52)
        df_inner['weekend'] = df_inner['create_time'].dt.dayofweek.isin([5,6]).astype(int)

        _holidays = None
        if config["country"] == "uk":
            _holidays = holidays.UK()
        elif config["country"] == "us":
            _holidays = holidays.US()
        elif config["country"] == "mys":
            _holidays = holidays.MYS()

        if _holidays:
            df_inner['holidays'] = df_inner['create_time'].dt.date.apply(lambda x: x in _holidays).astype(int)
        else:
            df_inner['holidays'] = 0

        return df_inner

    df = df.copy()
    df['create_time'] = pd.to_datetime(df['create_time'])
    df = add_time_features(df)
    df["log_sale"] = np.log(df.sale + 1e-8)
    df["avg_sale_by_id"] = df.groupby("id")['sale'].transform("mean")

    df = df.sort_values('create_time').reset_index(drop=True)
    df['time_idx'] = df.index

    return df

def prepare_features(df, scaler=None):
    processed = df[config['dynamic_features']].copy()
    target_series = df[config['target']].copy()

    if scaler is None:
        scaler = StandardScaler()
        processed[config['numeric_features']] = scaler.fit_transform(processed[config['numeric_features']])
    else:
        processed[config['numeric_features']] = scaler.transform(processed[config['numeric_features']])

    return processed, target_series, scaler

def prepare_features_old(df):
    processed = df[config['dynamic_features'] + [config['target']]].copy()

    scaler = StandardScaler()
    processed[config['numeric_features']] = scaler.fit_transform(processed[config['numeric_features']])

    return processed, scaler

def train(_config):
    config.update(_config)
    data_path = config["dataset_path"]
    train = read_train_df(config)
    print(f"{data_path} train size: {len(train)}")

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': []
    }
    successful_predictions_count = 0.0

    for product_id, group in train.groupby('id'):
        group = group.sort_values('create_time')
        if len(group) < config['max_prediction_length'] * 2:
            continue

        processed_exog, y, scaler = prepare_features(group)
        y = group[config['target']]

        split_idx = -config['max_prediction_length']
        train_y = y.iloc[:split_idx]
        train_exog = processed_exog.iloc[:split_idx]
        val_exog = processed_exog.iloc[split_idx:]

        processed, scaler = prepare_features_old(group)
        train_data = processed.iloc[:split_idx]
        val_data = processed.iloc[split_idx:]

        try:
            if config["model"] == "arima":
                model = ARIMA(
                    order=config['arima_order'],
                    seasonal_order=(0, 0, 0, 0),
                    exogenous=train_exog,
                    trace=False,
                    error_action='raise'
                )
                model.fit(train_y)
                forecast = model.predict(
                    n_periods=config['max_prediction_length'],
                    exogenous=val_exog
                )
            elif config["model"] == "sarima":
                model = ARIMA(
                    order=config['arima_order'],
                    seasonal_order=config['sarima_seasonal_order'],
                    exogenous=train_exog,
                    trace=False,
                    error_action='raise'
                )
                model.fit(train_y)
                forecast = model.predict(
                    n_periods=config['max_prediction_length'],
                    exogenous=val_exog
                )
            else:
                raise(f"error model: {config['model']}")

            y_true = y.iloc[split_idx:].values
            metrics['mae'].append(mean_absolute_error(y_true, forecast))
            metrics['mse'].append(mean_squared_error(y_true, forecast))
            metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))

            print(f"Product {product_id} | MAE: {metrics['mae'][-1]:.4f}")
            successful_predictions_count += 1
        except Exception as e:
            print(f"Product {product_id} train failed: {str(e)}")

    # 汇总最终指标
    print(f"config: {_config}")
    print("\n===== Final Metrics =====")
    print(f"AVG MAE: {sum(metrics['mae']) / successful_predictions_count:.4f}±{np.std(np.array(metrics['mae'])):.4f}")
    print(f"AVG MSE: {sum(metrics['mse']) / successful_predictions_count:.4f}±{np.std(np.array(metrics['mse'])):.4f}")
    print(f"AVG RMSE: {sum(metrics['rmse']) / successful_predictions_count:.4f}±{np.std(np.array(metrics['rmse'])):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="traditional_model")
    parser.add_argument("--model", type=str, default="cnn", help="model")
    parser.add_argument("--dataset_path", type=str, default="US_8903_Picture_cleaned.csv", help="dataset_path")
    parser.add_argument("--max_encoder_length", type=int, default=30, help="max_encoder_length")
    parser.add_argument("--max_prediction_length", type=int, default=2, help="max_prediction_length")

    args = parser.parse_args()
    train({
        "model": args.model,
        "dataset_path": args.dataset_path,
        "max_encoder_length": args.max_encoder_length,
        "max_prediction_length": args.max_prediction_length,
        })