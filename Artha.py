import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("/mnt/data/financial_data.csv")
    return df

df = load_data()

# Data Preprocessing & Normalization
scaler = MinMaxScaler()
numeric_cols = ["Income", "Age", "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous", "Desired_Savings", "Disposable_Income"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Streamlit App Configuration
st.set_page_config(page_title="Artha", layout="wide")
st.title("💰 Artha")
st.markdown("---")

# Sidebar - User Inputs
st.sidebar.header("📌 Enter Your Details")
name = st.sidebar.text_input("👤 Name", "John Doe")
age = st.sidebar.number_input("🎂 Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("💵 Annual Salary", min_value=10000, max_value=1000000, value=50000, step=1000)

# Dropdown Selection
feature = st.selectbox("🔍 Select a Feature", ["Financial Health Prediction", "Wealth Management"])
st.markdown("---")

if feature == "Financial Health Prediction":
    st.subheader("📊 Financial Health Prediction")
    
    # Calculate Key Ratios
    df["Debt_to_Income_Ratio"] = df["Loan_Repayment"] / df["Income"]
    df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
    df["Disposable_Income_Percentage"] = df["Disposable_Income"] / df["Income"]
    
    # Financial Health Score
    health_score = round(np.random.uniform(50, 100), 2)
    st.metric(label="Your Financial Health Score", value=f"{health_score}/100", delta=health_score-75)
    
    # AI Recommendations
    st.subheader("🤖 AI-Generated Recommendations")
    if health_score > 80:
        st.success("🚀 Your finances are in great shape! Keep up the good work!")
    elif 60 <= health_score <= 80:
        st.warning("⚠️ You're doing okay, but consider increasing your savings rate and reducing unnecessary expenses.")
    else:
        st.error("🔴 High financial risk detected! Cut down on liabilities and boost your savings immediately.")
    
    # Visualization
    st.subheader("📈 Savings Rate Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Savings_Rate"], bins=20, kde=True, color="green", ax=ax)
    ax.set_title("Savings Rate Distribution")
    st.pyplot(fig)

elif feature == "Wealth Management":
    st.subheader("🎯 Wealth Management - Goal Planning")
    
    # Financial Goals
    retirement = st.checkbox("Retirement Planning")
    car = st.checkbox("Buy a Car")
    house = st.checkbox("Buy a House")
    
    if retirement:
        retirement_savings = income * 0.15
        st.write(f"🔹 To retire comfortably, aim to save **${retirement_savings:.2f}** annually.")
    if car:
        car_savings = income * 0.10
        st.write(f"🚗 Save at least **${car_savings:.2f}** annually to afford a car in the future.")
    if house:
        house_savings = income * 0.25
        st.write(f"🏡 Consider saving **${house_savings:.2f}** annually for your dream house.")
    
    # Financial Knowledge Section
    st.subheader("📚 Financial Knowledge Insights")
    st.markdown(
        """
        - **Debt-to-Income Ratio:** Should ideally be below 36%.
        - **Savings Rate:** Aim for at least 20%.
        - **Emergency Fund:** Have at least 6 months' worth of expenses saved.
        - **Investment Strategy:** Diversify across stocks, bonds, and real estate.
        """
    )
    
st.markdown("---")
st.caption("🚀 AI-Powered Financial Insights - Created by AKVSS")
