# AI-Fitness: Intelligent Fitness & Nutrition Recommendation System

## Project Overview
AI-Fitness is a machine learning-powered web application that provides personalized fitness and nutrition recommendations. It leverages predictive models to estimate daily caloric intake, time to fitness goals, diet plans, and workout routines based on user inputs.

## Features
- Predict daily caloric needs using Linear Regression models
- Estimate time to achieve fitness goals
- Generate personalized diet and workout plans using Random Forest classifiers
- Real-time BMI and BMR calculations
- Responsive web interface built with Flask and Jinja2 templates
- End-to-end data preprocessing and model training pipeline

## Technologies Used
- Python 3.x
- Flask (Web framework)
- scikit-learn (Machine learning)
- pandas, NumPy (Data processing)
- HTML5, CSS3, JavaScript (Frontend)
- Faker (Synthetic data generation)

## Project Structure
- `app/`: Flask web application source code, templates, and static files
- `model/`: Serialized machine learning models (.pkl files)
- `data/`: Dataset files and preprocessing artifacts
- `scripts/`: Data preprocessing, model training, and evaluation scripts
- `run.py`: Entry point to start the Flask application
- `requirements.txt`: Python dependencies

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd ML-Fitness
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage
- Enter your age, height, weight, gender, and fitness goal on the input form.
- Submit to receive personalized predictions for caloric intake, diet plan, workout plan, and estimated time to reach your goal.
- View detailed results on the results page.

## Data Science Workflow
- Synthetic fitness data generated using Faker library
- Data preprocessing includes feature engineering (BMI, BMR) and scaling
- Models trained using Linear Regression and Random Forest classifiers
- Models saved as pickle files for production use

## Future Enhancements
- Add user authentication and profile management
- Integrate real-time activity tracking via wearable devices
- Expand diet and workout plan options with more granular recommendations
- Deploy on cloud platforms for scalability


