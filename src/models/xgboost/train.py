import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import matplotlib as plt
from utils import *
import pandas as pd
from xgboost import XGBRegressor

model_name = 'xgboost'
stage = 'train'

X, y = load_data('train')
X_values = X.values
y_values = y.values

param_grid = load_params(model_name)

base_model = XGBRegressor(random_state=42, verbosity=0)

if param_grid:
    model = perform_grid_search(base_model, param_grid, X_values, y_values)
else:
    model = base_model.fit(X_values, y_values)

y_pred = model.predict(X_values)
metrics = compute_metrics(y, y_pred)
save_metrics(metrics, model_name, stage)

save_model(model, model_name, scaler=None)

if hasattr(model, 'feature_importances_'):
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    imp_df.to_csv(f'models/{model_name}/feature_importance.csv', index=False)
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()  # чтобы самый важный признак был сверху
    plt.tight_layout()
    plt.savefig(f'models/{model_name}/feature_importance.png', dpi=150)

print(f"{model_name} {stage} completed.")