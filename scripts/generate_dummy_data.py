import pandas as pd
import numpy as np
import os

# Define the path to save the CSV file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
csv_path = os.path.join(data_dir, 'dummy_fitness_data.csv')

# Define the number of samples
num_samples = 100

# Generate dummy data
names = [f"Person_{i}" for i in range(num_samples)]
ages = np.random.randint(18, 60, size=num_samples)
genders = np.random.choice(['Male', 'Female'], size=num_samples)
heights = np.random.randint(150, 200, size=num_samples)
weights = np.random.randint(50, 100, size=num_samples)
bmi = weights / ((heights / 100) ** 2)
bmr = [10 * weight + 6.25 * height - 5 * age + (5 if gender == 'Male' else -161) for weight, height, age, gender in zip(weights, heights, ages, genders)]
goals = np.random.choice(['Lose', 'Gain', 'Maintain'], size=num_samples)
daily_caloric_intake = np.random.randint(1800, 3000, size=num_samples)
time_to_goal = np.random.randint(60, 180, size=num_samples)

# Generate new columns
recommended_caloric_intake = [1800 if goal == 'Lose' else 2500 if goal == 'Gain' else 2200 for goal in goals]
diet_plans = np.random.choice([
    'Low-carb diet: Breakfast - Eggs, Lunch - Chicken Salad, Dinner - Grilled Fish',
    'High-protein diet: Breakfast - Protein Shake, Lunch - Steak, Dinner - Cottage Cheese',
    'Balanced diet: Breakfast - Oatmeal, Lunch - Quinoa Salad, Dinner - Vegetable Stir Fry'
], size=num_samples)
workout_plans = np.random.choice([
    'Cardio: Running, Cycling, Swimming',
    'Weightlifting: Squats, Deadlifts, Bench Press',
    'Mixed: Cardio + Weightlifting, Yoga, HIIT'
], size=num_samples)

# Create a DataFrame
df = pd.DataFrame({
    'Name': names,
    'Age': ages,
    'Gender': genders,
    'Height_cm': heights,
    'Weight_kg': weights,
    'BMI': bmi,
    'BMR': bmr,
    'Goal': goals,
    'Daily_Caloric_Intake': daily_caloric_intake,
    'Time_to_Goal': time_to_goal,
    'Recommended_Caloric_Intake': recommended_caloric_intake,
    'Diet_Plan': diet_plans,
    'Workout_Plan': workout_plans
})

# Save the DataFrame to a CSV file
os.makedirs(data_dir, exist_ok=True)
df.to_csv(csv_path, index=False)

print(f"Dummy fitness data saved to '{csv_path}'")