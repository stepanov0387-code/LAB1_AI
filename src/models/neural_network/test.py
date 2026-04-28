import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from utils import load_data, load_model, compute_metrics, save_metrics, plot_predictions, plot_residuals

model_name = 'neural_network'
stage = 'test'

model, scaler = load_model(model_name, need_scaler=True)

X_test, y_test = load_data('test')
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

metrics = compute_metrics(y_test, y_pred)
save_metrics(metrics, model_name, stage)

plot_predictions(y_test, y_pred, model_name, stage)
plot_residuals(y_test, y_pred, model_name, stage)

print(f"{model_name} {stage} completed.")