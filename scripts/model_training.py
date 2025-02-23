import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load preprocessed data
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed'))
X_preprocessed_path = os.path.join(data_dir, 'X_preprocessed.csv')
y_caloric_path = os.path.join(data_dir, 'y_caloric.csv')
y_time_path = os.path.join(data_dir, 'y_time.csv')
y_recommended_caloric_path = os.path.join(data_dir, 'y_recommended_caloric.csv')
y_diet_plan_path = os.path.join(data_dir, 'y_diet_plan.csv')
y_workout_plan_path = os.path.join(data_dir, 'y_workout_plan.csv')

X = pd.read_csv(X_preprocessed_path)
y_caloric = pd.read_csv(y_caloric_path).values.ravel()
y_time = pd.read_csv(y_time_path).values.ravel()
y_recommended_caloric = pd.read_csv(y_recommended_caloric_path).values.ravel()
y_diet_plan = pd.read_csv(y_diet_plan_path).values.ravel()
y_workout_plan = pd.read_csv(y_workout_plan_path).values.ravel()

# Train caloric intake model
caloric_model = LinearRegression()
caloric_model.fit(X, y_caloric)

# Train time to goal model
time_model = LinearRegression()
time_model.fit(X, y_time)

# Train recommended caloric intake model
recommended_caloric_model = LinearRegression()
recommended_caloric_model.fit(X, y_recommended_caloric)

# Train diet plan model
diet_plan_model = RandomForestClassifier()
diet_plan_model.fit(X, y_diet_plan)

# Train workout plan model
workout_plan_model = RandomForestClassifier()
workout_plan_model.fit(X, y_workout_plan)

# Save the models
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, 'caloric_model.pkl'), 'wb') as f:
    pickle.dump(caloric_model, f)

with open(os.path.join(model_dir, 'time_model.pkl'), 'wb') as f:
    pickle.dump(time_model, f)

with open(os.path.join(model_dir, 'recommended_caloric_model.pkl'), 'wb') as f:
    pickle.dump(recommended_caloric_model, f)

with open(os.path.join(model_dir, 'diet_plan_model.pkl'), 'wb') as f:
    pickle.dump(diet_plan_model, f)

with open(os.path.join(model_dir, 'workout_plan_model.pkl'), 'wb') as f:
    pickle.dump(workout_plan_model, f)

print("Models trained and saved successfully.")