
import pandas as pd
import subprocess, sys
from sklearn.metrics import mean_squared_error, r2_score

# Auto-install required packages
for pkg in ['pandas', 'xgboost']:
    subprocess.call([sys.executable, "-m", "pip", "install", pkg])

import xgboost as xgb

df = pd.read_csv("JAIR_model_ready_dataset_2012_2024.csv")

features = ['Year', 'Female_Literacy', 'Urban_pct', 'Obesity_pct',
            'Tobacco_use_pct', 'Alcohol_use_pct', 'Screening_pct', 'Beds_per_1000']
target = 'AAR'

train = df[df['Year'] < 2024]
test = df[df['Year'] == 2024]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

model = xgb.XGBRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n✅ XGBoost Results:")
print("Test RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("Test R²:", r2_score(y_test, y_pred))
