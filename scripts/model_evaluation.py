import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Define the path to the preprocessed data and models
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data/preprocessed')
model_dir = os.path.join(base_dir, 'model')
X_path = os.path.join(data_dir, 'X_preprocessed.csv')
y_caloric_path = os.path.join(data_dir, 'y_caloric.csv')
y_time_path = os.path.join(data_dir, 'y_time.csv')

# Load the preprocessed data
X_test = pd.read_csv(X_path)
y_test_caloric = pd.read_csv(y_caloric_path)
y_test_time = pd.read_csv(y_time_path)

# Load the models
with open(os.path.join(model_dir, 'caloric_model.pkl'), 'rb') as f:
    model_caloric = pickle.load(f)

with open(os.path.join(model_dir, 'time_model.pkl'), 'rb') as f:
    model_time = pickle.load(f)

# Make predictions
y_pred_caloric = model_caloric.predict(X_test)
y_pred_time = model_time.predict(X_test)

# Evaluate the models
mae_caloric = mean_absolute_error(y_test_caloric, y_pred_caloric)
r2_caloric = r2_score(y_test_caloric, y_pred_caloric)

mae_time = mean_absolute_error(y_test_time, y_pred_time)
r2_time = r2_score(y_test_time, y_pred_time)

# Print the evaluation metrics
print(f'Caloric Model - Mean Absolute Error (MAE): {mae_caloric}')
print(f'Caloric Model - R-squared (R²): {r2_caloric}')
print(f'Time Model - Mean Absolute Error (MAE): {mae_time}')
print(f'Time Model - R-squared (R²): {r2_time}')