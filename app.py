import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("🏦 Customer Churn Prediction System")
st.markdown("""
Predict whether a customer is likely to churn using a machine learning model.
This app also provides risk levels and business insights.(Fill the details in the sidebar, click on left top arrow )
""")

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("churn_model.pkl")

# -------------------------------
# Sidebar Inputs (Cleaner UI)
# -------------------------------
st.sidebar.header("🧾 Customer Details")

credit_score = st.sidebar.number_input("Credit Score", 300, 900, 600)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -------------------------------
# Feature Engineering (same as training)
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
if st.button("🔮 Predict Churn"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    # Risk Levels
    if prob >= 0.7:
        st.error(f"🔴 High Risk of Churn ({prob:.2f})")
    elif prob >= 0.4:
        st.warning(f"🟡 Medium Risk of Churn ({prob:.2f})")
    else:
        st.success(f"🟢 Low Risk of Churn ({prob:.2f})")

    # Probability bar
    st.progress(float(prob))

    # -------------------------------
    # Business Insight
    # -------------------------------
    st.subheader("💰 Business Insight")

    if prob >= 0.7:
        st.write("👉 Immediate retention action recommended (offers, calls, engagement)")
    elif prob >= 0.4:
        st.write("👉 Monitor customer and consider soft engagement strategies")
    else:
        st.write("👉 Customer is stable, no immediate action needed")

    # -------------------------------
    # Explainability (Simple)
    # -------------------------------
    st.subheader("🔍 Key Drivers")

    reasons = []

    if is_active == 0:
        reasons.append("Customer is inactive (high churn risk)")

    if age > 45:
        reasons.append("Customer belongs to higher age group")

    if balance > 100000:
        reasons.append("High balance may indicate switching behavior")

    if num_products == 1:
        reasons.append("Low product engagement")

    if geography == "Germany":
        reasons.append("Geographical churn pattern observed")

    if len(reasons) > 0:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("No strong churn indicators detected")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit | End-to-End ML Project with Business Insights") 
