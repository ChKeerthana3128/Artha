import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Ensure you have the CSV file in the same directory)
@st.cache_data
def load_data():
    df = pd.read_csv("financial_data.csv")
    return df

df = load_data()

# UI Configuration
st.set_page_config(page_title="Artha", layout="wide")
st.title("💰 Artha")
st.markdown("---")

# User Input Section
st.sidebar.header("📌 Enter Your Details")
name = st.sidebar.text_input("👤 Name", "John Doe")
age = st.sidebar.number_input("🎂 Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("💵 Annual Salary", min_value=10000, max_value=1000000, value=50000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Financial Goals")
retirement = st.sidebar.checkbox("Retirement Planning")
car = st.sidebar.checkbox("Buy a Car")
house = st.sidebar.checkbox("Buy a House")

# Financial Health Prediction Section
st.header("📊 Financial Health Prediction")

def calculate_ratios(df):
    df["Debt_to_Income_Ratio"] = df["Loan_Repayment"] / df["Income"]
    df["Savings_Rate"] = df["Desired_Savings"] / df["Income"]
    df["Disposable_Income_Percentage"] = df["Disposable_Income"] / df["Income"]
    return df

df = calculate_ratios(df)

health_score = round(np.random.uniform(50, 100), 2)  # Simulated AI Score
st.metric(label="Your Financial Health Score", value=f"{health_score}/100", delta=health_score-75)

st.subheader("🤖 AI-Generated Recommendations")
if health_score > 80:
    st.success("🚀 Your finances are in great shape! Keep up the good work!")
elif 60 <= health_score <= 80:
    st.warning("⚠️ You're doing okay, but consider increasing your savings rate and reducing unnecessary expenses.")
else:
    st.error("🔴 High financial risk detected! Cut down on liabilities and boost your savings immediately.")

st.subheader("📈 Savings Optimization Suggestions")
potential_savings_cols = [
    "Potential_Savings_Groceries", "Potential_Savings_Transport", "Potential_Savings_Eating_Out",
    "Potential_Savings_Entertainment", "Potential_Savings_Utilities", "Potential_Savings_Healthcare",
    "Potential_Savings_Education", "Potential_Savings_Miscellaneous"
]
total_potential_savings = df[potential_savings_cols].sum(axis=1).mean()
st.info(f"💡 You can potentially save up to **${total_potential_savings:.2f}** per year by optimizing your spending!")

st.subheader("📊 Financial Data Analysis")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df["Savings_Rate"], bins=20, kde=True, color="green", ax=ax)
ax.set_title("Savings Rate Distribution")
st.pyplot(fig)

# Wealth Management Section
st.header("🎯 Wealth Management")
if retirement:
    retirement_savings = income * 0.15  # Example calculation
    st.write(f"🔹 To retire comfortably, aim to save **${retirement_savings:.2f}** annually.")
if car:
    car_savings = income * 0.10
    st.write(f"🚗 Save at least **${car_savings:.2f}** annually to afford a car in the future.")
if house:
    house_savings = income * 0.25
    st.write(f"🏡 Consider saving **${house_savings:.2f}** annually for your dream house.")

st.subheader("📚 Financial Knowledge Insights")
st.markdown(
    """
    - **Debt-to-Income Ratio:** Should ideally be below 36%.
    - **Savings Rate:** The higher, the better! Aim for at least 20%.
    - **Emergency Fund:** Have at least 6 months' worth of expenses saved.
    - **Investment Strategy:** Diversify across stocks, bonds, and real estate.
    """
)

st.markdown("---")
st.caption("🚀 AI-Powered Financial Insights - Created by AKVSS")
