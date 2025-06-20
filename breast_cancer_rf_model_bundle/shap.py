import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data = pd.read_csv("final_breast_cancer_dataset_for_modeling.csv")
features = ['Year', 'Female_Literacy', 'Urban_pct', 'Obesity_pct',
            'Tobacco_use_pct', 'Alcohol_use_pct', 'Screening_pct', 'Beds_per_1000']
train = data[data['Year'] < 2016]
X = train[features]
y = train['AAR']

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
plt.savefig("shap_feature_importance_rf.png")
