import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('my_model_ann.h5')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Failure Predictor", page_icon="💔", layout="centered")
st.markdown("<h1 style='text-align: center; color: crimson;'>💔 Heart Failure Death Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Provide the patient’s medical data to predict the likelihood of a fatal heart failure event.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📋 Instructions")
st.sidebar.write("""
- Input real patient values.
- The model will predict the risk of death due to heart failure.
- Values must match the format used during training.
""")

# Input UI
st.subheader("🩺 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 50)
    anaemia = st.radio("Anaemia", ["Yes", "No"])
    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=200)
    diabetes = st.radio("Diabetes", ["Yes", "No"])
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 35)
    high_blood_pressure = st.radio("High Blood Pressure", ["Yes", "No"])

with col2:
    platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=250000.0, step=1000.0, format="%.2f")
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.1, step=0.1, format="%.2f")
    serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=100, max_value=150, value=137)
    sex = st.radio("Sex", ["Male", "Female"])
    smoking = st.radio("Smoking", ["Yes", "No"])
    time = st.slider("Follow-up Period (days)", 0, 300, 130)

# Encoding inputs
def encode(value):
    if value in ["Yes", "Male"]:
        return 1
    else:
        return 0

features = np.array([[
    age,
    encode(anaemia),
    creatinine_phosphokinase,
    encode(diabetes),
    ejection_fraction,
    encode(high_blood_pressure),
    platelets,
    serum_creatinine,
    serum_sodium,
    encode(sex),
    encode(smoking),
    time
]])

# Scale
features_scaled = scaler.transform(features)

# Prediction
if st.button("🔍 Predict Risk"):
    prediction = model.predict(features_scaled)[0][0]
    st.markdown("---")
    if prediction > 0.5:
        st.error(f"🚨 High Risk of Death (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ Low Risk of Death (Probability: {prediction:.2f})")
