
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🧠 ASD Prediction App")

# Input fields
age = st.slider("Age", 1, 100, 25)
gender = st.selectbox("Gender", ['male', 'female'])
ethnicity = st.selectbox("Ethnicity", ['White-European', 'Latino', 'Others'])

# A1 - A10 screening answers (binary)
a_scores = [st.selectbox(f"Q{i+1} Answer", [0, 1]) for i in range(10)]

# Other binary questions
jundice = st.selectbox("Born with jaundice?", [0, 1])
family_history = st.selectbox("Family member with ASD?", [0, 1])
used_app_before = st.selectbox("Used screening app before?", [0, 1])
who_completed_test = st.selectbox("Test completed by self?", [0, 1])

# Encode gender and ethnicity manually
gender_encoded = 1 if gender == 'male' else 0
ethnicity_encoded = 0 if ethnicity == 'White-European' else (1 if ethnicity == 'Latino' else 2)

# Final input
input_data = np.array([[age, gender_encoded, ethnicity_encoded] + a_scores + [jundice, family_history, used_app_before, who_completed_test]])

# Scale
input_scaled = scaler.transform(input_data)

if st.button("Predict ASD"):
    prediction = model.predict(input_scaled)[0]
    result = "Positive for ASD" if prediction == 1 else "Negative for ASD"
    st.success(f"Prediction: {result}")
