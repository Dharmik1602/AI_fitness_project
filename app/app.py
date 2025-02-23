from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__, template_folder='templates')

# Load models
model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
with open(os.path.join(model_dir, 'caloric_model.pkl'), 'rb') as f:
    caloric_model = pickle.load(f)

with open(os.path.join(model_dir, 'time_model.pkl'), 'rb') as f:
    time_model = pickle.load(f)

with open(os.path.join(model_dir, 'recommended_caloric_model.pkl'), 'rb') as f:
    recommended_caloric_model = pickle.load(f)

with open(os.path.join(model_dir, 'diet_plan_model.pkl'), 'rb') as f:
    diet_plan_model = pickle.load(f)

with open(os.path.join(model_dir, 'workout_plan_model.pkl'), 'rb') as f:
    workout_plan_model = pickle.load(f)

# Load the fitted preprocessor
preprocessor_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed', 'preprocessor.pkl')
with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / (height / 100) ** 2

# Function to calculate BMR using the Mifflin-St Jeor Equation
def calculate_bmr(age, weight, height, gender):
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

@app.route('/')
def landing_page():
    return render_template('landing.html')

@app.route('/info')
def info_page():
    return render_template('info.html')

@app.route('/app')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data['Age'] = int(data['Age'])
    data['Height_cm'] = float(data['Height_cm'])
    data['Weight_kg'] = float(data['Weight_kg'])
    
    # Calculate BMI and BMR
    bmi = calculate_bmi(data['Weight_kg'], data['Height_cm'])
    bmr = calculate_bmr(data['Age'], data['Weight_kg'], data['Height_cm'], data['Gender'])
    data['BMI'] = bmi
    data['BMR'] = bmr

    # Create a DataFrame from the input data with the correct column names
    df = pd.DataFrame([data], columns=['Age', 'Height_cm', 'Weight_kg', 'BMI', 'BMR', 'Gender', 'Goal'])

    # Preprocess input data
    X_input = preprocessor.transform(df)
    
    # Make predictions
    caloric_prediction = caloric_model.predict(X_input)
    time_prediction = time_model.predict(X_input)
    recommended_caloric_prediction = recommended_caloric_model.predict(X_input)
    diet_plan_prediction = diet_plan_model.predict(X_input)
    workout_plan_prediction = workout_plan_model.predict(X_input)

    # Render the result template with predictions
    return render_template('result.html', 
                           daily_caloric_intake=caloric_prediction[0], 
                           time_to_goal=time_prediction[0],
                           recommended_caloric_intake=recommended_caloric_prediction[0],
                           diet_plan=diet_plan_prediction[0],
                           workout_plan=workout_plan_prediction[0],
                           bmi=bmi,
                           bmr=bmr)

if __name__ == '__main__':
    app.run(debug=True)