import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# User Input for Wealth Management
st.title("💰 AI-Based Wealth Management Dashboard 🚀")
st.markdown("### 📊 Plan Your Financial Future with Smart Insights")

name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=18, max_value=100, step=1)
salary = st.number_input("Enter your current annual salary:", min_value=0, step=1000)

if name and age and salary:
    st.subheader(f"Welcome, {name}! Let's plan your financial future.")
    retirement_age = st.number_input("At what age do you plan to retire?", min_value=age, max_value=100, step=1)
    goal = st.selectbox("What are you planning for?", ["Retirement", "Buying a Car", "Buying a House"])
    
    # Load dataset directly from code
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
    df['Predicted_Savings'] = df['Disposable_Income'] * np.random.uniform(0.8, 1.2)
    
    # Predictive Insights
    years_until_retirement = max(0, retirement_age - age)
    suggested_savings = (salary * 0.15) * years_until_retirement if goal == "Retirement" else salary * 0.25
    
    st.subheader("📌 Financial Planning Insights")
    st.write(f"For your goal: **{goal}**, you should aim to save approximately **₹{suggested_savings:,.2f}** over the next {years_until_retirement} years.")
    
    st.subheader("📚 Financial Knowledge")
    st.markdown(
        "A strong financial plan includes budgeting, investing, and saving for long-term goals. "
        "Start by allocating a portion of your salary to different financial buckets: "
        "essential expenses, discretionary spending, and savings. The key to financial success "
        "is consistency in saving and making informed investment choices."
    )
    
    # Generate Alerts & Recommendations
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
    
    df['Predictive_Alert'] = df.apply(generate_predictive_alert, axis=1)
    df['Recommendations'] = df.apply(generate_recommendations, axis=1)
    
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
