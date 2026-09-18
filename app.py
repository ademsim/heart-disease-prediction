import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Heart Disease Predictor")
st.write("Hasta ölçümlerini gir, model kalp hastalığı riskini tahmin etsin.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 54)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda v: "Male" if v == 1 else "Female")
    cp = st.selectbox("Chest pain type (cp)", options=[1, 2, 3, 4])
    trestbps = st.slider("Resting blood pressure (trestbps)", 80, 200, 130)
    chol = st.slider("Cholesterol (chol)", 100, 600, 246)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", options=[0, 1])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.slider("Max heart rate (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise induced angina (exang)", options=[0, 1])
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
    slope = st.selectbox("Slope", options=[1, 2, 3])
    ca = st.selectbox("Number of major vessels (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thal", options=["normal", "fixed", "reversible"])

thal_normal = 1 if thal == "normal" else 0
thal_reversible = 1 if thal == "reversible" else 0

if st.button("Tahmin et"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal_normal": thal_normal,
        "thal_reversible": thal_reversible,
    }])

    prediction = model.predict(input_df)[0]
    label = "Kalp hastalığı riski VAR" if prediction == 1 else "Kalp hastalığı riski YOK"
    st.success(f"Tahmin: **{label}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        st.write(f"Hastalık olasılığı: **{proba[1]:.2%}**")
