# run_random_forest.py

import pandas as pd
import joblib
import os
import subprocess
import sys

# Auto-install required packages if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ["pandas", "scikit-learn", "joblib"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

from sklearn.metrics import mean_squared_error, r2_score

# Load model and dataset
model = joblib.load("random_forest_breast_model.pkl")
data = pd.read_csv("final_breast_cancer_dataset_for_modeling.csv")

# Define features and target
features = [
    'Year', 'Female_Literacy', 'Urban_pct', 'Obesity_pct',
    'Tobacco_use_pct', 'Alcohol_use_pct', 'Screening_pct', 'Beds_per_1000',
    'CR_female', 'TR_female', 'Cumulative_Risk_Female',
    'Sex_Ratio', 'Area_sqkm', 'MV_Percent', 'Top_Cancer_Breast'
]
target = 'Breast_Cancer_Rate'


# Prepare test data (Year 2016)
test = data[data['Year'] == 2016]
X_test = test[features]
y_test = test[target]

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n✅ Random Forest Model Evaluation:")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R²:   {r2:.2f}\n")
