import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("🏦 Customer Churn Prediction System")
st.markdown("""
Predict whether a customer is likely to churn using a machine learning model.
Includes risk levels, business insights, financial impact, and explainability.
""")

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("churn_model.pkl")

# Extract model & preprocessor for SHAP
rf_model = model.named_steps['model']
preprocessor = model.named_steps['preprocessing']

# -------------------------------
# Sidebar Inputs
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
# Strategy Mode (instead of threshold slider)
# -------------------------------
mode = st.sidebar.selectbox(
    "Retention Strategy",
    ["Balanced", "Aggressive Retention", "Cost Saving"]
)

if mode == "Aggressive Retention":
    threshold = 0.3
elif mode == "Balanced":
    threshold = 0.5
else:
    threshold = 0.7

st.sidebar.caption(f"Using threshold = {threshold}")

# -------------------------------
# Feature Engineering
# -------------------------------
balance_salary_ratio = balance / (salary + 1)
balance_salary_ratio_log = np.log1p(balance_salary_ratio)

if age < 30:
    age_group = "Young"
elif age < 45:
    age_group = "Adult"
elif age < 60:
    age_group = "Senior"
else:
    age_group = "Old"

if credit_score < 500:
    credit_group = "Low"
elif credit_score < 650:
    credit_group = "Medium"
elif credit_score < 750:
    credit_group = "Good"
else:
    credit_group = "Excellent"

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
    prediction = int(prob > threshold)

    st.subheader("📊 Prediction Result")

    # -------------------------------
    # Confidence Bands
    # -------------------------------
    if 0.4 <= prob <= 0.6:
        st.info(f"⚠️ Uncertain Prediction Zone ({prob:.2f})")
    elif prob > 0.7:
        st.error(f"🔴 High Risk of Churn ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"🟡 Medium Risk of Churn ({prob:.2f})")
    else:
        st.success(f"🟢 Low Risk of Churn ({prob:.2f})")

    st.write(f"📌 Final Decision (threshold={threshold}): {'Churn' if prediction else 'No Churn'}")
    st.progress(float(prob))

    # -------------------------------
    # 💰 Business Value
    # -------------------------------
    st.subheader("💰 Financial Impact")

    avg_customer_value = 50000
    retention_cost = 5000

    revenue_at_risk = prob * avg_customer_value
    net_gain = revenue_at_risk - retention_cost

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Revenue at Risk", f"₹{revenue_at_risk:,.0f}")

    with col2:
        st.metric("Potential Net Gain", f"₹{net_gain:,.0f}")

    st.caption("Estimated using churn probability — not guaranteed outcome.")

    # -------------------------------
    # 💡 Business Insight
    # -------------------------------
    st.subheader("💡 Business Insight")

    if prediction == 1:
        st.write("👉 Customer likely to churn — retention action recommended")
    else:
        st.write("👉 Customer likely to stay — no immediate action needed")

    # -------------------------------
    # 🔍 SHAP Explainability
    # -------------------------------
    if prediction == 1:
        st.subheader("🔍 Why is this customer at risk?")

        # Get SHAP values
        X_transformed = preprocessor.transform(input_data)
        explainer = shap.Explainer(rf_model)
        shap_values = explainer(X_transformed)

        shap_vals = shap_values.values

        # Fix shape
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[0, :, 1]
        else:
            shap_vals = shap_vals[0]

        feature_names = preprocessor.get_feature_names_out()

        # Create dataframe
        feature_importance = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_vals
        })

        feature_importance["AbsImpact"] = np.abs(feature_importance["Impact"])
        feature_importance = feature_importance.sort_values(by="AbsImpact", ascending=False)

        top_features = feature_importance.head(3)

        # -------------------------------
        # Convert to simple language
        # -------------------------------
        messages = []

        for feature in top_features["Feature"]:

            if "IsActiveMember" in feature:
                messages.append("Customer is inactive → higher churn risk")

            elif "Age" in feature:
                messages.append("Customer age group is linked with higher churn")

            elif "NumOfProducts" in feature:
                messages.append("Low product usage → low engagement")

            elif "BalanceSalaryRatio" in feature:
                messages.append("High balance relative to salary → possible switching behavior")

            elif "Geography_Germany" in feature:
                messages.append("Customers from this region tend to churn more")

            elif "CreditScore" in feature:
                messages.append("Credit score influences churn likelihood")

        # Remove duplicates
        messages = list(set(messages))

        if messages:
            for msg in messages:
                st.write(f"👉 {msg}")
        else:
            st.write("No strong churn indicators detected")
        st.subheader("🛠️ Recommended Actions")

        actions = []

        for feature in top_features["Feature"]:

            if "IsActiveMember" in feature:
                actions.append("Re-engage customer with personalized offers or notifications")

            elif "NumOfProducts" in feature:
                actions.append("Cross-sell additional products to increase engagement")

            elif "BalanceSalaryRatio" in feature:
                actions.append("Provide premium support or financial advisory to retain high-value customer")

            elif "Geography_Germany" in feature:
                actions.append("Apply region-specific retention strategies")

            elif "Age" in feature:
                actions.append("Offer tailored services based on customer life stage")

            elif "CreditScore" in feature:
                actions.append("Provide financial incentives or credit-related benefits")

        actions = list(set(actions))

        if actions:
            for act in actions:
                st.write(f"👉 {act}")
        else:
            st.write("No specific action required")
# -------------------------------
# Batch Prediction
# -------------------------------
st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df_batch = pd.read_csv(uploaded_file)

    required_cols = ['CreditScore','Geography','Gender','Age','Tenure',
                     'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

    missing = [col for col in required_cols if col not in df_batch.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df_batch['BalanceSalaryRatio'] = df_batch['Balance'] / (df_batch['EstimatedSalary'] + 1)
    df_batch['BalanceSalaryRatio_log'] = np.log1p(df_batch['BalanceSalaryRatio'])

    df_batch['AgeGroup'] = pd.cut(df_batch['Age'], bins=[18,30,45,60,100],
                                 labels=['Young','Adult','Senior','Old'])

    df_batch['CreditScoreGroup'] = pd.cut(df_batch['CreditScore'],
                                         bins=[300,500,650,750,900],
                                         labels=['Low','Medium','Good','Excellent'])

    df_batch = df_batch.drop(columns=['BalanceSalaryRatio'], errors='ignore')
    df_batch = df_batch.drop(columns=['Surname','CustomerId','RowNumber'], errors='ignore')

    preds = model.predict_proba(df_batch)[:,1]

    df_batch['Churn_Probability'] = preds
    df_batch['Prediction'] = (df_batch['Churn_Probability'] > threshold).astype(int)

    st.subheader("📊 Batch Results")
    st.dataframe(df_batch.head())

    csv = df_batch.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit | End-to-End ML System with Explainability & Business Impact")
