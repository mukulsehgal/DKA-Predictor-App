import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load CSV data
df = pd.read_csv('DKA with BMI.csv')

# Features and target
features = ['HGB_ALC_RESULT', 'PO2', 'PCO2', 'BASE_EXCESS', 'FIO2_VENOUS', 'O2_SAT_CALC_VENOUS',
            'SODIUM', 'POTASSIUM', 'CHLORIDE', 'GLUCOSE', 'BUN', 'CREATININE', 'ANION_GAP', 'HC03',
            'WHITE_BLOOD_CELL', 'RED_BLOOD_CELL', 'HEMATOCRIT', 'MEAN_CELL_VOL', 'MEAN_CELL_HEMOGLOBIN',
            'MEAN_CELL_HEMOGLOBIN_CONCENT', 'RDW_SD', 'MEAN_PLATELET_VOL', 'PLATELET', 'HEMOGLOBIN']

target = 'PH'

# Data preprocessing
df_model = df[features + [target]].apply(pd.to_numeric, errors='coerce').dropna()

X = df_model[features]
y = df_model[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
joblib.dump(model, 'DKA_pH_Predictor_Local.pkl')
print("Model trained and saved as 'DKA_pH_Predictor_Local.pkl'")
