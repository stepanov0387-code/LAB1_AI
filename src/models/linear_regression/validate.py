import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *

model_name = 'linear_regression'
stage = 'valid'

model, scaler = load_model(model_name, need_scaler=True)
X, y = load_data('valid')
X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)
metrics = compute_metrics(y, y_pred)
save_metrics(metrics, model_name, stage)

plot_predictions(y, y_pred, model_name, stage)
plot_residuals(y, y_pred, model_name, stage)

print(f"{model_name} {stage} completed.")