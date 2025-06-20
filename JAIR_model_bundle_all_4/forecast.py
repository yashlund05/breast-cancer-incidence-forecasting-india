import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Load your final dataset
df = pd.read_csv("JAIR_model_ready_dataset_2012_2024.csv")

# Define the features used in the model
features = ['Year', 'Female_Literacy', 'Urban_pct', 'Obesity_pct',
            'Tobacco_use_pct', 'Alcohol_use_pct', 'Screening_pct', 'Beds_per_1000']

# Get the latest available values per region (2024)
latest = df.sort_values("Year").groupby("Region").last().reset_index()
latest["Year"] = 2025  # Set year to forecast

X_2025 = latest[features]

# Train the XGBoost model on all available data
X = df[features]
y = df["AAR"]
model = xgb.XGBRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

# Predict AAR for 2025
latest["Predicted_AAR_2025"] = model.predict(X_2025)

# Sort by AAR for plotting
latest_sorted = latest.sort_values("Predicted_AAR_2025", ascending=False)

# Plot the results
plt.figure(figsize=(12, 6))
plt.barh(latest_sorted["Region"], latest_sorted["Predicted_AAR_2025"], color="skyblue")
plt.xlabel("Predicted AAR (2025)")
plt.title("Forecasted Breast Cancer AAR for 2025 - All 28 PBCRs")
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("forecast_2025_bar_chart.png")
plt.show()
