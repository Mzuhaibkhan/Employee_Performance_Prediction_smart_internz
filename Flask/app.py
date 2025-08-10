# Import the libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('../Training files/gwp.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submit')
def submit_page():
    return render_template('submit.html', prediction="Please use the predict page to make a prediction first.")

@app.route('/submit', methods=['POST'])
def pred():
    try:
        # Retrieve all the values from the HTML form using POST request
        quarter = request.form['quarter']
        department = request.form['department']
        day = request.form['day']
        team = request.form['team']
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        over_time = float(request.form['over_time'])
        incentive = float(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = float(request.form['idle_men'])
        no_of_style_change = float(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])
        month = float(request.form['month'])
        
        if department.replace(" ", "").lower() == 'finishing':
            department_encoded = 0 
        else:
            department_encoded = 1  
        
        quarter_encoded = 0 if quarter == 'Quarter1' else 1 if quarter == 'Quarter2' else 2 if quarter == 'Quarter3' else 3
        
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_encoded = day_map.get(day, 0)
        team_encoded = hash(team) % 100 
        input_features = np.array([[quarter_encoded, department_encoded, day_encoded, team_encoded, 
                                   targeted_productivity, smv, over_time, incentive, idle_time, 
                                   idle_men, no_of_style_change, no_of_workers, month]])
        
        prediction = model.predict(input_features)
        
        prediction_value = round(prediction[0], 4)
        
        return render_template('submit.html', prediction=prediction_value)
        
    except Exception as e:
        error_message = f"Error in prediction: {str(e)}"
        return render_template('submit.html', prediction=error_message)

# Main Function
if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
