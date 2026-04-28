import pandas as pd
import numpy as np
import json
import pickle
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow import keras  # добавлено

def load_data(dataset_type='train', model_name=None, need_scaler=False):
    """
    Загружает X и y для указанного типа датасета (train/valid/test).
    Если need_scaler=True, загружает и возвращает scaler.
    """
    X = pd.read_csv(f'data/processed/X_{dataset_type}.csv')
    y = pd.read_csv(f'data/processed/y_{dataset_type}.csv').squeeze()
    if need_scaler:
        with open(f'models/{model_name}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return X, y, scaler
    return X, y

def save_metrics(metrics, model_name, stage):
    """Сохраняет метрики в JSON."""
    os.makedirs(f'metrics/{model_name}', exist_ok=True)
    with open(f'metrics/{model_name}/{stage}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def save_model(model, model_name, scaler=None):
    os.makedirs(f'models/{model_name}', exist_ok=True)
    
    # Если это KerasRegressor, сохраняем внутреннюю Keras-модель
    if hasattr(model, 'model_'):
        keras_model = model.model_
        keras_model.save(f'models/{model_name}/model.keras')
    else:
        # Для обычных sklearn-моделей оставляем pickle
        with open(f'models/{model_name}/model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    if scaler:
        with open(f'models/{model_name}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

def load_model(model_name, need_scaler=False):
    keras_path = f'models/{model_name}/model.keras'
    pkl_path = f'models/{model_name}/model.pkl'
    
    if os.path.exists(keras_path):
        model = keras.models.load_model(keras_path)
    elif os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Model file not found for {model_name}")
    
    if need_scaler:
        with open(f'models/{model_name}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return model

def compute_metrics(y_true, y_pred):
    """Вычисляет MSE, MAE, RMSE, R2."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}

def plot_predictions(y_true, y_pred, model_name, stage, save_path=None):
    """Строит график предсказаний vs фактические значения."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: {stage} Predictions')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'models/{model_name}/{stage}_pred_vs_actual.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name, stage, save_path=None):
    """Строит график остатков."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: {stage} Residuals')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'models/{model_name}/{stage}_residuals.png')
    plt.close()

def perform_grid_search(model, param_grid, X, y, cv=5, scoring='neg_mean_squared_error'):
    """Выполняет GridSearchCV и возвращает лучшую модель."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    print(f"Best params: {grid.best_params_}")
    return grid.best_estimator_

def load_params(model_name):
    """Загружает сетку параметров из params.yaml для указанной модели."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['models'][model_name].get('search', {})