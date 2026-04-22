# -bank-churn-APP

## 📌 Project Overview
This project predicts whether a bank customer is likely to churn (leave the bank) using machine learning.

The goal is to help businesses identify high-risk customers and take proactive retention actions.

---

## 🎯 Business Problem
Customer churn is a major issue for banks and financial institutions.

- Losing customers = loss of revenue  
- Retaining customers = lower cost than acquiring new ones  

👉 This model helps identify customers at risk of leaving so targeted actions can be taken.

---

## 📊 Dataset
- 10,000 customer records  
- Features include:
  - Credit Score
  - Geography
  - Age
  - Balance
  - Number of Products
  - Activity Status
  - Estimated Salary

Target:
- `Exited` → 1 (Churn), 0 (Retained)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights:
- Older customers are more likely to churn  
- Inactive customers churn ~2x more than active customers  
- High balance customers show higher churn tendency  

---

## 🧪 Statistical Analysis

- **T-test** → Age is statistically significant  
- **Chi-square test** → Activity strongly impacts churn  
- **Confidence Interval** → Validated age trends  

---

## ⚙️ Feature Engineering

Created meaningful features:
- AgeGroup  
- CreditScoreGroup  
- BalanceSalaryRatio (log transformed)

---

## 🤖 Model Development

Models tested:
- Logistic Regression  
- Random Forest  

Final Model:
👉 **Random Forest (Tuned)**

---

## 🔧 Model Optimization

- Hyperparameter tuning using GridSearchCV  
- Optimized for **Recall** (important for churn detection)  
- Threshold tuning applied  

---

## 📈 Model Performance

- Recall (Churn): **73%**  
- ROC-AUC Score: **0.85**  
- Lift (Top Decile): **~4.7**

👉 Model identifies high-risk customers ~5x better than random selection

---

## 🧠 Key Business Insights

- Inactive users are most likely to churn  
- High-value customers (high balance) need retention focus  
- Targeting top 20% high-risk customers can capture majority churn  

---

## 🖥️ Deployment

Built an interactive web app using:
- :Streamlit

Features:
- User input form  
- Real-time prediction  
- Risk classification (Low / Medium / High)  
- Business recommendations  

---

## 🚀 How to Run Locally
https://churnapp1.streamlit.app/

## Author
Animesh Sandhu
