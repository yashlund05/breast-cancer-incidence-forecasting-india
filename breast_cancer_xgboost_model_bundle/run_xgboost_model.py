# run_xgboost_model.py

import pandas as pd
import xgboost as xgb
import subprocess
import sys
from sklearn.metrics import mean_squared_error, r2_score

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["pandas", "xgboost", "scikit-learn"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

model = xgb.XGBRegressor()
model.load_model("xgboost_breast_model.json")
data = pd.read_csv("final_breast_cancer_dataset_for_modeling.csv")

features = [
    'Year',
    'Female_Literacy',
    'Urban_pct',
    'Obesity_pct',
    'Tobacco_use_pct',
    'Alcohol_use_pct',
    'Screening_pct',
    'Beds_per_1000'
]
target = 'AAR'

test = data[data['Year'] == 2016]
X_test = test[features]
y_test = test[target]

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n✅ XGBoost Model Evaluation:")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²:   {r2:.2f}\n")
