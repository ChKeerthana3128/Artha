import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset with correct path
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/financial_data.csv")
    return df

df = load_data()

# Data Preprocessing
def preprocess_data(df):
    df.fillna(0, inplace=True)
    numeric_cols = ['Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous',
                    'Desired_Savings', 'Disposable_Income']
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df["Debt_to_Income_Ratio"] = df["Loan_Repayment"] / df["Income"]
    df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
    df["Disposable_Income_Percentage"] = df["Disposable_Income"] / df["Income"]
    return df

df = preprocess_data(df)

# UI Configuration
st.set_page_config(page_title="AI-Based Financial Dashboard", layout="wide")
st.title("💰 AI-Based Financial Health & Wealth Management Dashboard 🚀")
st.markdown("---")

# Sidebar: User Input
st.sidebar.header("📌 Enter Your Details")
name = st.sidebar.text_input("👤 Name", "John Doe")
age = st.sidebar.number_input("🎂 Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("💵 Annual Salary", min_value=10000, max_value=1000000, value=50000, step=1000)
st.sidebar.markdown("---")

# Financial Goals
st.sidebar.subheader("🎯 Financial Goals")
retirement_age = st.sidebar.number_input("At what age do you plan to retire?", min_value=age, max_value=100, step=1)
goal = st.sidebar.selectbox("What are you planning for?", ["Retirement", "Buying a Car", "Buying a House"])

# Financial Health Score Calculation
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
suggested_savings = (income * 0.15) * years_until_retirement if goal == "Retirement" else income * 0.25

st.subheader("📊 Financial Data Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Savings_Rate"], bins=20, kde=True, color="green", ax=ax)
ax.set_title("Savings Rate Distribution")
st.pyplot(fig)

fig = px.histogram(df, x='Financial_Health_Score', nbins=20, title='📈 Financial Health Score Distribution')
st.plotly_chart(fig)

fig2 = px.scatter(df, x='Income', y='Predicted_Savings', color='Financial_Health_Score', title='💡 Income vs Predicted Savings')
st.plotly_chart(fig2)

# Sidebar Dropdowns for Insights & Recommendations
with st.sidebar.expander("📌 Wealth Management Insights & Recommendations"):
    st.write(f"For your goal: **{goal}**, you should aim to save approximately **₹{suggested_savings:,.2f}** over the next {years_until_retirement} years.")
    st.markdown(
        "A strong financial plan includes budgeting, investing, and saving for long-term goals. "
        "Start by allocating a portion of your salary to different financial buckets: "
        "essential expenses, discretionary spending, and savings. The key to financial success "
        "is consistency in saving and making informed investment choices."
    )

with st.sidebar.expander("🔍 Financial Health Insights & Recommendations"):
    for index, row in df.iterrows():
        st.write(f"**👤 User {index + 1}:**")
        st.write(f"- 💳 Financial Health Score: {row['Financial_Health_Score']:.2f}")
        st.write(f"- {row['Predictive_Alert']}")
        st.write(f"- 🎯 Recommendations: {row['Recommendations']}")
        st.write("---")

st.markdown("---")
st.caption("🚀 AI-Powered Financial Insights - Created by AKVSS")
