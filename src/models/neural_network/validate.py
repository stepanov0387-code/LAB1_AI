import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from utils import load_data, load_model, compute_metrics, save_metrics, plot_predictions, plot_residuals

model_name = 'neural_network'
stage = 'valid'

model, scaler = load_model(model_name, need_scaler=True)

X_val, y_val = load_data('valid')
X_val_scaled = scaler.transform(X_val)

y_pred = model.predict(X_val_scaled)

metrics = compute_metrics(y_val, y_pred)
save_metrics(metrics, model_name, stage)

plot_predictions(y_val, y_pred, model_name, stage)
plot_residuals(y_val, y_pred, model_name, stage)

print(f"{model_name} {stage} completed.")