import streamlit as st
import joblib
import numpy as np

# Load Models
ph_model = joblib.load("DKA_pH_Predictor_Local.pkl")
wbc_model = joblib.load("DKA_WBC_Predictor_With_pH.pkl")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Auto-train models if not found
if not os.path.exists("DKA_pH_Predictor_Local.pkl") or not os.path.exists("DKA_WBC_Predictor_With_pH.pkl"):
    st.write("Training models... please wait (first-time setup)")

    # Load dataset
    df = pd.read_csv('DKA with BMI.csv')

    # Common Features
    features = ['HGB_ALC_RESULT', 'PO2', 'PCO2', 'BASE_EXCESS', 'FIO2_VENOUS', 'O2_SAT_CALC_VENOUS',
                'SODIUM', 'POTASSIUM', 'CHLORIDE', 'GLUCOSE', 'BUN', 'CREATININE', 'ANION_GAP', 'HC03',
                'RED_BLOOD_CELL', 'HEMATOCRIT', 'MEAN_CELL_VOL', 'MEAN_CELL_HEMOGLOBIN',
                'MEAN_CELL_HEMOGLOBIN_CONCENT', 'RDW_SD', 'MEAN_PLATELET_VOL', 'PLATELET', 'HEMOGLOBIN']

    # --- Train pH Model ---
    target_ph = 'PH'
    df_ph = df[features + [target_ph]].apply(pd.to_numeric, errors='coerce').dropna()
    X_ph = df_ph[features]
    y_ph = df_ph[target_ph]
    X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_ph, y_ph, test_size=0.3, random_state=42)
    ph_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ph_model.fit(X_train_ph, y_train_ph)
    joblib.dump(ph_model, 'DKA_pH_Predictor_Local.pkl')

    # --- Train WBC Model with pH ---
    features_wbc = ['PH'] + features
    target_wbc = 'WHITE_BLOOD_CELL'
    df_wbc = df[features_wbc + [target_wbc]].apply(pd.to_numeric, errors='coerce').dropna()
    X_wbc = df_wbc[features_wbc]
    y_wbc = df_wbc[target_wbc]
    X_train_wbc, X_test_wbc, y_train_wbc, y_test_wbc = train_test_split(X_wbc, y_wbc, test_size=0.3, random_state=42)
    wbc_model = RandomForestRegressor(n_estimators=100, random_state=42)
    wbc_model.fit(X_train_wbc, y_train_wbc)
    joblib.dump(wbc_model, 'DKA_WBC_Predictor_With_pH.pkl')

    st.success("Models trained successfully!")
else:
    ph_model = joblib.load("DKA_pH_Predictor_Local.pkl")
    wbc_model = joblib.load("DKA_WBC_Predictor_With_pH.pkl")

st.title("DKA pH & WBC Prediction Tool")
st.write("Input patient lab values to predict severity of acidosis (pH) and WBC count")

# Input fields (shared)
hgba1c = st.number_input('HbA1c (%)', min_value=5.0, max_value=15.0, value=10.0)
po2 = st.number_input('PO2 (mmHg)', min_value=20.0, max_value=100.0, value=45.0)
pco2 = st.number_input('PCO2 (mmHg)', min_value=10.0, max_value=50.0, value=30.0)
base_excess = st.number_input('Base Excess', min_value=-30.0, max_value=0.0, value=-15.0)
fio2_venous = st.number_input('FiO2 Venous (%)', min_value=21.0, max_value=100.0, value=21.0)
o2_sat = st.number_input('Venous O2 Saturation (%)', min_value=40.0, max_value=100.0, value=75.0)
sodium = st.number_input('Sodium (mmol/L)', min_value=120.0, max_value=160.0, value=135.0)
potassium = st.number_input('Potassium (mmol/L)', min_value=2.0, max_value=6.0, value=4.5)
chloride = st.number_input('Chloride (mmol/L)', min_value=90.0, max_value=120.0, value=105.0)
glucose = st.number_input('Glucose (mg/dL)', min_value=100.0, max_value=1000.0, value=400.0)
bun = st.number_input('BUN (mg/dL)', min_value=5.0, max_value=50.0, value=18.0)
creatinine = st.number_input('Creatinine (mg/dL)', min_value=0.3, max_value=2.0, value=1.0)
anion_gap = st.number_input('Anion Gap', min_value=5.0, max_value=30.0, value=20.0)
hco3 = st.number_input('Bicarbonate (mmol/L)', min_value=5.0, max_value=30.0, value=10.0)
rbc = st.number_input('RBC (x10^9/L)', min_value=3.0, max_value=6.0, value=5.0)
hematocrit = st.number_input('Hematocrit (%)', min_value=20.0, max_value=50.0, value=40.0)
mcv = st.number_input('MCV (fL)', min_value=70.0, max_value=100.0, value=85.0)
mch = st.number_input('MCH (pg)', min_value=20.0, max_value=35.0, value=28.0)
mchc = st.number_input('MCHC (g/dL)', min_value=30.0, max_value=37.0, value=33.5)
rdw_sd = st.number_input('RDW-SD (fL)', min_value=35.0, max_value=50.0, value=42.0)
mpv = st.number_input('Mean Platelet Volume (fL)', min_value=7.0, max_value=14.0, value=10.5)
platelet = st.number_input('Platelet Count (x10^9/L)', min_value=100.0, max_value=700.0, value=350.0)
hemoglobin = st.number_input('Hemoglobin (g/dL)', min_value=8.0, max_value=18.0, value=13.0)

if st.button("Predict pH and WBC"):
    # Input array for pH Prediction
    ph_input = np.array([[hgba1c, po2, pco2, base_excess, fio2_venous, o2_sat, sodium, potassium, chloride,
                          glucose, bun, creatinine, anion_gap, hco3, rbc, hematocrit, mcv, mch,
                          mchc, rdw_sd, mpv, platelet, hemoglobin]])

    predicted_ph = ph_model.predict(ph_input)[0]
    st.subheader(f"Predicted pH: {predicted_ph:.3f}")

    if predicted_ph < 7.1:
        st.error("Severe Acidosis (pH < 7.1)")
    elif predicted_ph < 7.25:
        st.warning("Moderate Acidosis")
    else:
        st.info("Mild Acidosis")

    # Input array for WBC Prediction (include predicted pH)
    wbc_input = np.array([[predicted_ph, hgba1c, po2, pco2, base_excess, fio2_venous, o2_sat, sodium, potassium, chloride,
                           glucose, bun, creatinine, anion_gap, hco3, rbc, hematocrit, mcv, mch,
                           mchc, rdw_sd, mpv, platelet, hemoglobin]])

    predicted_wbc = wbc_model.predict(wbc_input)[0]
    st.subheader(f"Predicted WBC Count: {predicted_wbc:.2f} x10^9/L")

    if predicted_wbc > 20:
        st.error("High Leukocytosis (WBC > 20 x10^9/L)")
    elif predicted_wbc > 11:
        st.warning("Mild Leukocytosis")
    else:
        st.info("Normal WBC Range")

