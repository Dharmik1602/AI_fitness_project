import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

# Define the path to the CSV file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
csv_path = os.path.join(data_dir, 'dummy_fitness_data.csv')

# Load the data
df = pd.read_csv(csv_path)

# Print column names for verification
print("Columns in the dataset:", df.columns)

# Define preprocessing for numerical and categorical features
numerical_features = ['Age', 'Height_cm', 'Weight_kg', 'BMI', 'BMR']
categorical_features = ['Gender', 'Goal']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Drop only the columns that exist in the DataFrame
columns_to_drop = ['Name', 'Diet_Plan', 'Time_to_Goal', 'Daily_Caloric_Intake', 'Recommended_Caloric_Intake', 'Diet_Plan', 'Workout_Plan']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

# Apply preprocessing
X = df.drop(existing_columns_to_drop, axis=1)
y_caloric = df['Daily_Caloric_Intake']
y_time = df['Time_to_Goal']
y_recommended_caloric = df['Recommended_Caloric_Intake']
y_diet_plan = df['Diet_Plan']
y_workout_plan = df['Workout_Plan']

# Fit and transform the features
X_preprocessed = preprocessor.fit_transform(X)

# Save the fitted preprocessor
preprocessed_data_dir = os.path.join(data_dir, 'preprocessed')
os.makedirs(preprocessed_data_dir, exist_ok=True)
preprocessor_path = os.path.join(preprocessed_data_dir, 'preprocessor.pkl')
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)

# Save the preprocessed data to CSV files
pd.DataFrame(X_preprocessed).to_csv(os.path.join(preprocessed_data_dir, 'X_preprocessed.csv'), index=False)
y_caloric.to_csv(os.path.join(preprocessed_data_dir, 'y_caloric.csv'), index=False)
y_time.to_csv(os.path.join(preprocessed_data_dir, 'y_time.csv'), index=False)
y_recommended_caloric.to_csv(os.path.join(preprocessed_data_dir, 'y_recommended_caloric.csv'), index=False)
y_diet_plan.to_csv(os.path.join(preprocessed_data_dir, 'y_diet_plan.csv'), index=False)
y_workout_plan.to_csv(os.path.join(preprocessed_data_dir, 'y_workout_plan.csv'), index=False)

print(f"Preprocessed data and preprocessor saved to '{preprocessed_data_dir}'")