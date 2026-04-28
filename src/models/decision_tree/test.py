import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils import *

model_name = 'decision_tree'
stage = 'test'

model = load_model(model_name, need_scaler=False)

X, y = load_data('test')

y_pred = model.predict(X.values)

metrics = compute_metrics(y, y_pred)
save_metrics(metrics, model_name, stage)

plot_predictions(y, y_pred, model_name, stage)
plot_residuals(y, y_pred, model_name, stage)

print(f"{model_name} {stage} completed.")