import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def calculate_financial_health_score(row):
    debt_to_income = row['Loan_Repayment'] / row['Income'] if row['Income'] > 0 else 1
    savings_rate = row['Desired_Savings'] / row['Income'] if row['Income'] > 0 else 0
    discretionary_spending = (row['Eating_Out'] + row['Entertainment'] + row['Miscellaneous']) / row['Income']
    
    score = 100 - (debt_to_income * 40 + discretionary_spending * 30 - savings_rate * 30)
    return max(0, min(100, score))

def generate_predictive_alert(row):
    if row['Predicted_Savings'] < row['Desired_Savings']:
        return "⚠️ Warning! Your current spending habits may not meet your savings goal."
    return "✅ Your financial trajectory is stable. Keep up the good work!"

def generate_recommendations(row):
    recs = []
    if row['Predicted_Savings'] < row['Desired_Savings']:
        if row['Eating_Out'] > 0:
            recs.append("🍽️ Reduce dining out expenses.")
        if row['Entertainment'] > 0:
            recs.append("🎭 Cut back on entertainment spending.")
        if row['Miscellaneous'] > 0:
            recs.append("💡 Re-evaluate miscellaneous expenses.")
    return " | ".join(recs) if recs else "🎯 Your spending habits are well-balanced."

# Load dataset directly from code
df = pd.read_csv("financial_data.csv")

# Process Data
df['Financial_Health_Score'] = df.apply(calculate_financial_health_score, axis=1)
df['Predicted_Savings'] = df['Disposable_Income'] * np.random.uniform(0.8, 1.2)
df['Predictive_Alert'] = df.apply(generate_predictive_alert, axis=1)
df['Recommendations'] = df.apply(generate_recommendations, axis=1)

# Streamlit UI
st.title("💰 AI-Based Financial Health Dashboard 🚀")
st.markdown("### 📊 Personalized Insights for Better Financial Decisions")

st.subheader("🔍 Financial Summary")
st.write(df[['Income', 'Financial_Health_Score', 'Predicted_Savings', 'Predictive_Alert', 'Recommendations']])

fig = px.histogram(df, x='Financial_Health_Score', nbins=20, title='📈 Financial Health Score Distribution')
st.plotly_chart(fig)

fig2 = px.scatter(df, x='Income', y='Predicted_Savings', color='Financial_Health_Score', title='💡 Income vs Predicted Savings')
st.plotly_chart(fig2)

st.subheader("📌 Insights & Recommendations")
for index, row in df.iterrows():
    st.write(f"**👤 User {index + 1}:**")
    st.write(f"- 💳 Financial Health Score: {row['Financial_Health_Score']:.2f}")
    st.write(f"- {row['Predictive_Alert']}")
    st.write(f"- 🎯 Recommendations: {row['Recommendations']}")
    st.write("---")
