import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("🏦 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn")

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("churn_model.pkl")

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -------------------------------
# -------------------------------

# Balance Salary Ratio (log)
balance_salary_ratio = balance / (salary + 1)
balance_salary_ratio_log = np.log1p(balance_salary_ratio)

# Age Group
if age < 30:
    age_group = "Young"
elif age < 45:
    age_group = "Adult"
elif age < 60:
    age_group = "Senior"
else:
    age_group = "Old"

# Credit Score Group
if credit_score < 500:
    credit_group = "Low"
elif credit_score < 650:
    credit_group = "Medium"
elif credit_score < 750:
    credit_group = "Good"
else:
    credit_group = "Excellent"

# -------------------------------
# Create Input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [salary],
    'AgeGroup': [age_group],
    'CreditScoreGroup': [credit_group],
    'BalanceSalaryRatio_log': [balance_salary_ratio_log]
})

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    # Risk levels
    if prob >= 0.7:
        st.error(f"⚠️ High Risk of Churn ({prob:.2f})")
    elif prob >= 0.4:
        st.warning(f"🟠 Medium Risk of Churn ({prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Churn ({prob:.2f})")

    # Progress bar
    st.progress(float(prob))

    # -------------------------------
    # Business Insight
    # -------------------------------
    st.subheader("📈 Business Insight")

    if prob > 0.3:
        st.write("👉 Recommend retention strategy (offer, engagement, call)")
    else:
        st.write("👉 Customer is stable, no immediate action needed")

    # -------------------------------
    # Simple Explainability
    # -------------------------------
    st.subheader("🔍 Possible Reasons")

    reasons = []

    if is_active == 0:
        reasons.append("Customer is inactive")

    if age > 45:
        reasons.append("Customer belongs to higher age group")

    if balance > 100000:
        reasons.append("Customer has high balance (possible switching risk)")

    if num_products == 1:
        reasons.append("Customer has low product engagement")

    if len(reasons) > 0:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("No strong churn indicators detected")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.write("Built using Streamlit | ML Churn Prediction Project")