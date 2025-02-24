import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 🏡 **UI Configuration**
st.set_page_config(page_title="Artha", layout="wide")
st.title("Artha - AI-Based Financial Dashboard")

# 📌 **Load Dataset with Error Handling**
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("financial_data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Error: The dataset file 'financial_data.csv' is missing.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    
    # 🔹 **Data Preprocessing**
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

    # 🔹 **Train Multiple Linear Regression Model**
    X = df[['Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
            'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']]
    y = df['Desired_Savings']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    df['Predicted_Savings'] = model.predict(X)

    # 🏠 **Sidebar - User Input**
    st.sidebar.header("📌 Enter Your Details")
    name = st.sidebar.text_input("👤 Name", "John Doe")
    age = st.sidebar.number_input("🎂 Age", min_value=18, max_value=100, value=30)
    income = st.sidebar.number_input("💵 Annual Salary", min_value=10000, max_value=1000000, value=50000, step=1000)
    
    st.sidebar.markdown("---")

    # 🔹 **Sidebar Insights & Recommendations**
    with st.sidebar.expander("📊 Wealth Management Insights"):
        st.write("- Plan your financial goals effectively.")
        st.write("- Allocate savings wisely based on your income.")

    with st.sidebar.expander("💡 Financial Health Insights"):
        st.write("- Monitor your debt-to-income ratio.")
        st.write("- Optimize discretionary spending for better savings.")

    # 🎯 **Financial Goals**
    st.sidebar.subheader("🎯 Financial Goals")
    retirement_age = st.sidebar.number_input("At what age do you plan to retire?", min_value=age, max_value=100, step=1)
    goal = st.sidebar.selectbox("What are you planning for?", ["Retirement", "Buying a Car", "Buying a House"])

    # 🔹 **Apply Filters Button**
    apply_filters = st.sidebar.button("✅ Apply Filters")

    if apply_filters:
        # 🔹 **Financial Health Score Calculation**
        def calculate_financial_health_score(row):
            debt_to_income = row['Loan_Repayment'] / row['Income'] if row['Income'] > 0 else 1
            savings_rate = row['Desired_Savings'] / row['Income'] if row['Income'] > 0 else 0
            discretionary_spending = (row['Eating_Out'] + row['Entertainment'] + row['Miscellaneous']) / row['Income'] if row['Income'] > 0 else 0
            score = 100 - (debt_to_income * 40 + discretionary_spending * 30 - savings_rate * 30)
            return max(0, min(100, score))

        df['Financial_Health_Score'] = df.apply(calculate_financial_health_score, axis=1)

        # 🔹 **Predictive Insights**
        years_until_retirement = max(0, retirement_age - age)
        suggested_savings = (income * 0.15) * years_until_retirement if goal == "Retirement" else income * 0.25

        st.subheader("📌 Financial Planning Insights")
        st.write(f"For your goal: **{goal}**, you should aim to save approximately **₹{suggested_savings:,.2f}** over the next {years_until_retirement} years.")

        st.subheader("📚 Financial Knowledge")
        st.markdown(
            "A strong financial plan includes budgeting, investing, and saving for long-term goals. "
            "Start by allocating a portion of your salary to different financial buckets: "
            "essential expenses, discretionary spending, and savings. "
            "The key to financial success is consistency in saving and making informed investment choices."
        )

        # 📊 **Data Visualization with Table**
        st.subheader("📊 Financial Data Analysis")

        # 🔹 **Financial Health Score Distribution**
        st.subheader("📈 Financial Health Score Distribution")
        fig = px.histogram(df, x='Financial_Health_Score', nbins=20, title='📈 Financial Health Score Distribution')
        st.plotly_chart(fig)
        st.dataframe(df[['Income', 'Financial_Health_Score', 'Savings_Rate', 'Debt_to_Income_Ratio']]
                     .sort_values(by='Financial_Health_Score', ascending=False))

        # 🔹 **Income vs Predicted Savings**
        st.subheader("💡 Income vs Predicted Savings")
        fig2 = px.scatter(df, x='Income', y='Predicted_Savings', color='Financial_Health_Score', title='💡 Income vs Predicted Savings')
        st.plotly_chart(fig2)
        st.dataframe(df[['Income', 'Predicted_Savings', 'Disposable_Income_Percentage']]
                     .sort_values(by='Predicted_Savings', ascending=False))

        st.markdown("---")
        st.caption("🚀 AI-Powered Financial Insights - Created by AKVSS")
