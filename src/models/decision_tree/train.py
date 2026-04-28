import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

model_name = 'decision_tree'
stage = 'train'
X, y = load_data('train')

param_grid = load_params(model_name)

base_model = DecisionTreeRegressor()

if param_grid:
    model = perform_grid_search(base_model, param_grid, X.values, y.values)
else:
    model = base_model.fit(X.values, y.values)

y_pred = model.predict(X.values)
metrics = compute_metrics(y, y_pred)
save_metrics(metrics, model_name, stage)

save_model(model, model_name, scaler=None)

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, max_depth=3, fontsize=10)
plt.savefig(f'models/{model_name}/tree_nodes.png')
plt.close()

if hasattr(model, 'feature_importances_'):
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    imp_df.to_csv(f'models/{model_name}/feature_importance.csv', index=False)

print(f"{model_name} {stage} completed.")