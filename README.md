# Employee_Performance_Prediction_smart_internz
Project repository for a Machine Learning Approach for Employee Performance Prediction Model for the Virtual Internship for Smart Internz.

## 1. Overview
This project predicts garment worker (employee) productivity using machine learning. It includes:
- Dataset: Garments worker productivity data.
- Training Notebook: Builds and evaluates several regression models and saves the best model as `gwp.pkl`.
- Flask Web App: Serves a user interface for submitting feature inputs and viewing predicted productivity.

## 2. Repository Structure
```
Dataset/
  garments_worker_productivity.csv
Training files/
  Employee_Prediction.ipynb
  gwp.pkl                # Generated after running notebook (not tracked until created)
Flask/
  app.py                 # Flask backend (ensure present/updated)
  templates/
    home.html
    predict.html
    about.html
    submit.html
README.md
Documentation.md
```

## 3. Data Schema (Key Columns)
| Column | Description |
|--------|-------------|
| quarter | Quarter of year (Quarter1–Quarter4) |
| department | Production department (e.g., sweing, finishing) |
| day | Day of week |
| team | Team identifier (Team1, Team2, etc.) |
| targeted_productivity | Planned productivity (0–1) |
| smv | Standard Minute Value |
| over_time | Overtime minutes |
| incentive | Incentive value |
| idle_time | Idle minutes |
| idle_men | Number of idle workers |
| no_of_style_change | Style change count |
| no_of_workers | Total workers |
| month | Extracted from date |
| actual_productivity | TARGET (0–1) |

## 4. Model Training Workflow (Notebook: `Employee_Prediction.ipynb`)
1. Load dataset and inspect (EDA, correlations).
2. Clean / transform (drop unused columns, normalize department labels, derive `month`).
3. Encode categorical variables with `LabelEncoder` per column.
4. Split into train/test sets.
5. Train multiple regressors:
   - LinearRegression
   - RandomForestRegressor
   - XGBRegressor
6. Evaluate using R², MSE, MAE.
7. Select best model by R².
8. Persist best model to `gwp.pkl` using `pickle`.

## 5. Serialization Notes
Current workflow saves only the model, NOT the encoders. For production consistency you should also persist each `LabelEncoder` used, otherwise category-to-integer mappings must be reconstructed exactly.

## 6. Feature Order for Prediction
The model expects features in this order:
```
['quarter','department','day','team',
 'targeted_productivity','smv','over_time','incentive',
 'idle_time','idle_men','no_of_style_change','no_of_workers','month']
```

## 7. Flask Application Responsibilities
- Load `gwp.pkl` at startup.
- Render pages: `home`, `predict` (form), `about`, `submit` (result).
- Accept POST submission, preprocess inputs, predict, format output as percentage with 4 decimals.

### Example Minimal `app.py`
```
from flask import Flask, render_template, request
import pickle, numpy as np, os

MODEL_PATH = os.path.join('..', 'Training files', 'gwp.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

app = Flask(__name__)

quarter_map = {'Quarter1':0,'Quarter2':1,'Quarter3':2,'Quarter4':3}
dept_map = {'sweing':0,'finishing':1}
day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

def encode_team(team: str) -> int:
    return abs(hash(team)) % 100  # Placeholder; replace with persisted encoder

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/submit', methods=['POST'])
def submit():
    form = request.form
    try:
        quarter = quarter_map.get(form.get('quarter','Quarter1'), 0)
        department = dept_map.get(form.get('department','sweing'), 0)
        day = day_map.get(form.get('day','Monday'), 0)
        team = encode_team(form.get('team','Team1'))
        targeted_productivity = float(form.get('targeted_productivity',0))
        smv = float(form.get('smv',0))
        over_time = float(form.get('over_time',0))
        incentive = float(form.get('incentive',0))
        idle_time = float(form.get('idle_time',0))
        idle_men = float(form.get('idle_men',0))
        no_of_style_change = float(form.get('no_of_style_change',0))
        no_of_workers = float(form.get('no_of_workers',0))
        month = float(form.get('month',1))

        features = np.array([[quarter, department, day, team,
                              targeted_productivity, smv, over_time, incentive,
                              idle_time, idle_men, no_of_style_change, no_of_workers, month]])
        pred = model.predict(features)[0] * 100
        prediction = f"{pred:.4f}%"
    except Exception as exc:
        prediction = f"Error: {exc}"
    return render_template('submit.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
```

## 8. Environment Setup & Running the App
### Prerequisites
- Python 3.9+
- pip

### Create and Activate Virtual Environment (Windows PowerShell)
```
cd "c:\Vs Code\Github\Pycharm\Virual Internship"
python -m venv venv
venv\Scripts\Activate.ps1
```

### Install Dependencies
```
pip install flask pandas seaborn matplotlib scikit-learn xgboost
```

### (Optional) Re-train & Generate `gwp.pkl`
Open Jupyter / VS Code and run all cells in:
```
Training files/Employee_Prediction.ipynb
```
Confirm `Training files/gwp.pkl` exists.

### Run the Flask App
```
cd "Flask"
python app.py
```
Open: http://127.0.0.1:5000/

## 9. Improving Categorical Handling
To ensure reproducible predictions:
1. During training, after fitting each LabelEncoder, save it:
```
import pickle
with open('encoders.pkl','wb') as f:
    pickle.dump({'quarter':q_enc, 'department':d_enc, ...}, f)
```
2. Load in `app.py` and apply in identical order.
3. Replace hashing logic for `team` with real encoder lookup.

## 10. Troubleshooting
| Symptom | Likely Cause | Resolution |
|---------|--------------|-----------|
| ImportError xgboost | Package missing | pip install xgboost |
| ValueError shape mismatch | Feature order off | Use documented order |
| Wrong percentages | Encoders inconsistent | Persist & reuse encoders |
| App shows 0% always | Inputs not parsed (names mismatch) | Match form field names |
| Slow startup | Large model | Optimize / reduce estimators |

## 11. Future Enhancements
- Persist full preprocessing pipeline (`Pipeline` in scikit-learn).
- Add input validation & flash messages.
- Containerize with Docker.
- Add unit tests (pytest) for prediction route.
- Introduce CI workflow (GitHub Actions) to retrain and validate metrics.
- Add authentication (if needed) for internal use.

## 12. License / Attribution
Verify licensing/usage rights for the garments productivity dataset before distribution.

## 13. Changelog (Manual)
- Initial notebook creation & model training.
- Added Flask interface.
- Enhanced UI (Tailwind CSS) & percentage formatting.

## 14. Contact
For questions open an issue or contact the repository owner.

Done.
