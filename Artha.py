import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("financial_data.csv")

# Data Preprocessing
scaler = MinMaxScaler()
numeric_cols = ['Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous',
                'Desired_Savings', 'Disposable_Income']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Calculate Financial Health Score
def calculate_financial_health_score(row):
    debt_to_income = row['Loan_Repayment'] / row['Income'] if row['Income'] > 0 else 1
    savings_rate = row['Desired_Savings'] / row['Income'] if row['Income'] > 0 else 0
    discretionary_spending = (row['Eating_Out'] + row['Entertainment'] + row['Miscellaneous']) / row['Income'] if row['Income'] > 0 else 0
    
    score = 100 - (debt_to_income * 40 + discretionary_spending * 30 - savings_rate * 30)
    return max(0, min(100, score))

df['Financial_Health_Score'] = df.apply(calculate_financial_health_score, axis=1)

# Train Model
features = numeric_cols
target = 'Financial_Health_Score'

X = df[features]
y = df[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.title("💰 AI-Based Financial Health Dashboard 🚀")
st.markdown("### 📊 Personalized Insights for Better Financial Decisions")

# Sidebar Inputs
st.sidebar.header("📌 Enter Your Financial Details")
user_input = {}
for col in numeric_cols:
    user_input[col] = st.sidebar.number_input(f"{col.replace('_', ' ')}", min_value=0.0, step=0.1)

data_input = pd.DataFrame([user_input])
data_input[numeric_cols] = scaler.transform(data_input[numeric_cols])
predicted_score = model.predict(data_input)[0]

# Display Prediction
st.sidebar.subheader("🧮 Predicted Financial Health Score")
st.sidebar.write(f"📊 Your Financial Health Score: **{predicted_score:.2f}**")

# Dataset Preview
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# Visualization
fig = px.histogram(df, x='Financial_Health_Score', nbins=20, title='📈 Financial Health Score Distribution')
st.plotly_chart(fig)

fig2 = px.scatter(df, x='Income', y='Financial_Health_Score', color='Financial_Health_Score', title='💡 Income vs Financial Health Score')
st.plotly_chart(fig2)
