import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_name = 'linear_regression'
stage = 'train'

X, y = load_data('train')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

param_grid = load_params(model_name)
base_model = LinearRegression()

if param_grid:
    model = perform_grid_search(base_model, param_grid, X_scaled, y)
else:
    model = base_model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
metrics = compute_metrics(y, y_pred)
save_metrics(metrics, model_name, stage)
save_model(model, model_name, scaler)

plt.figure()
plt.hist(model.coef_, bins=20, edgecolor='black')
plt.title('Linear Regression Weights')
plt.savefig(f'models/{model_name}/weights_hist.png')
plt.close()

print(f"{model_name} {stage} completed.")