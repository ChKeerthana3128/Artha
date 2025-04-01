import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import requests
from fpdf import FPDF
import io
import os
import tempfile
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="💰 Artha", layout="wide", initial_sidebar_state="expanded")

# Simulated Investment Dataset
investment_data = pd.DataFrame({
    "Company": ["Reliance Industries", "HDFC Bank", "Bajaj Finance", "SBI Bluechip Fund",
                "Paytm", "Zomato", "Bitcoin", "Ethereum"],
    "Category": ["Large Cap", "Large Cap", "Medium Cap", "Medium Cap",
                 "Low Cap", "Low Cap", "Crypto", "Crypto"],
    "Min_Invest": [1000, 500, 1500, 500, 2000, 2000, 5000, 3000],
    "Risk": ["Medium", "Low", "Medium", "Medium", "High", "High", "High", "High"],
    "Goal": ["Wealth growth", "Emergency fund", "Future expenses", "Emergency fund",
             "Wealth growth", "Future expenses", "No specific goal", "Wealth growth"],
    "Expected_Return": [8.5, 6.0, 10.0, 7.5, 15.0, 14.0, 20.0, 18.0],
    "Volatility": [15.0, 10.0, 20.0, 12.0, 30.0, 28.0, 50.0, 45.0]
})
investment_data["Risk_Encoded"] = investment_data["Risk"].map({"Low": 0, "Medium": 1, "High": 2})
investment_data["Goal_Encoded"] = investment_data["Goal"].map({
    "Wealth growth": 0, "Emergency fund": 1, "Future expenses": 2, "No specific goal": 3
})

# Data Loading Functions
@st.cache_data
def load_stock_data(csv_path="NIFTY CONSUMPTION_daily_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Stock CSV not found! Please upload 'NIFTY CONSUMPTION_daily_data.csv'")
        return None
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['Date'].isnull().all():
            st.error("🚨 Invalid date format in stock data!")
            return None
        df = df[['Date', 'open', 'high', 'low', 'close', 'volume']].sort_values(by='Date').dropna()
        return df
    except Exception as e:
        st.error(f"🚨 Error loading stock data: {str(e)}")
        return None

@st.cache_data
def load_survey_data(csv_path="survey_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Survey CSV not found! Please upload 'survey_data.csv'.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]
        def parse_range(value):
            if pd.isna(value) or value in ["I don’t save", ""]:
                return 0
            if "Above" in value:
                return float(value.split("₹")[1].replace(",", "")) + 500
            if "₹" in value:
                bounds = value.split("₹")[1].split("-")
                if len(bounds) == 2:
                    return (float(bounds[0].replace(",", "")) + float(bounds[1].replace(",", ""))) / 2
                return float(bounds[0].replace(",", ""))
            return float(value)
        df["Income"] = df["How much pocket money or income do you receive per month (in ₹)?"].apply(parse_range)
        df["Essentials"] = df["How much do you spend monthly on essentials (e.g., food, transport, books)?"].apply(parse_range)
        df["Non_Essentials"] = df["How much do you spend monthly on non-essentials (e.g., entertainment, eating out)?"].apply(parse_range)
        df["Debt_Payment"] = df["If yes to debt, how much do you pay monthly (in ₹)?"].apply(parse_range)
        df["Savings"] = df["How much of your pocket money/income do you save each month (in ₹)?"].apply(parse_range)
        return df
    except Exception as e:
        st.error(f"🚨 Error loading survey data: {str(e)}")
        return None

@st.cache_data
def load_financial_data(csv_path="financial_data.csv"):
    if not os.path.exists(csv_path):
        st.error("🚨 Financial CSV not found! Please upload 'financial_data.csv'.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().replace('"', '') for col in df.columns]
        col_map = {col.lower(): col for col in df.columns}
        required_cols = ["income"]
        missing_cols = [col for col in required_cols if col not in col_map]
        if missing_cols:
            st.error(f"🚨 'financial_data.csv' is missing required columns: {', '.join(missing_cols)}")
            return None
        df = df.rename(columns={col_map["income"]: "income"})
        if "projected_savings" not in col_map:
            df["Projected_Savings"] = df["income"] * 0.2
            st.warning("⚠️ 'projected_savings' not found. Using 20% of income.")
        else:
            df = df.rename(columns={col_map["projected_savings"]: "Projected_Savings"})
        expense_cols = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport", "Healthcare", 
                       "Education", "Miscellaneous (Eating_Out,Entertainmentand Utilities)"]
        available_expense_cols = [col for col in expense_cols if col in df.columns]
        if available_expense_cols:
            df["Total_Expenses"] = df[available_expense_cols].sum(axis=1)
        else:
            df["Total_Expenses"] = 0
        return df
    except Exception as e:
        st.error(f"🚨 Error loading financial data: {str(e)}")
        return None

# Model Training Functions
@st.cache_resource
def train_stock_model(data):
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    X = data[['Day', 'Month', 'Year']]
    y = data['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training stock prediction model..."):
        model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

@st.cache_resource
def train_survey_model(survey_data):
    features = ["Income", "Essentials", "Non_Essentials", "Debt_Payment"]
    target = "Savings"
    X = survey_data[features].fillna(0)
    y = survey_data[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training savings prediction model..."):
        model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

@st.cache_resource
def train_retirement_model(financial_data):
    features = ["income", "Total_Expenses"]
    target = "Projected_Savings"
    X = financial_data[features].fillna(0)
    y = financial_data[target].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training retirement savings model..."):
        model.fit(X_train, y_train)
    return model, r2_score(y_test, model.predict(X_test))

@st.cache_resource
def train_investment_model(data):
    X = data[["Min_Invest", "Risk_Encoded", "Goal_Encoded", "Expected_Return", "Volatility"]]
    y = (data["Expected_Return"] / data["Volatility"]) * (1 - data["Risk_Encoded"] * 0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    with st.spinner("Training investment recommendation model..."):
        model.fit(X, y)
    return model

# Predictive and Utility Functions
def predict_savings(model, income, essentials, non_essentials, debt_payment):
    input_df = pd.DataFrame({
        "Income": [income],
        "Essentials": [ess timingsentials],
        "Non_Essentials": [non_essentials],
        "Debt_Payment": [debt_payment]
    })
    return model.predict(input_df)[0]

def predict_retirement_savings(model, income, expenses):
    input_df = pd.DataFrame({"income": [income], "Total_Expenses": [expenses]})
    return model.predict(input_df)[0]

def calculate_savings_goal(goal_amount, horizon_years):
    return goal_amount / (horizon_years * 12) if horizon_years > 0 else goal_amount

def forecast_retirement_savings(income, savings, years, growth_rate=5.0):
    wealth = savings
    monthly_savings = savings
    for _ in range(years * 12):
        wealth = wealth * (1 + growth_rate / 1200) + monthly_savings
    return wealth

def predict_investment_strategy(model, invest_amount, risk_tolerance, horizon_years, goals):
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    goal_map = {"Wealth growth": 0, "Emergency fund": 1, "Future expenses": 2, "No specific goal": 3}
    risk_encoded = risk_map[risk_tolerance]
    goal_encoded_list = [goal_map[goal] for goal in goals]
    
    input_data = investment_data[["Min_Invest", "Risk_Encoded", "Goal_Encoded", "Expected_Return", "Volatility"]].copy()
    input_data["Expected_Return"] = input_data["Expected_Return"] * (1 + horizon_years * 0.05)
    input_data["Volatility"] = input_data["Volatility"] * (1 - horizon_years * 0.02)
    
    scores = model.predict(input_data)
    investment_data["Suitability_Score"] = scores
    
    filtered = investment_data[
        (investment_data["Min_Invest"] <= invest_amount) &
        (investment_data["Risk_Encoded"] <= risk_encoded) &
        (investment_data["Goal_Encoded"].isin(goal_encoded_list) | 
         (investment_data["Goal_Encoded"] == goal_map["No specific goal"]))
    ]
    
    recommendations = {}
    for category in filtered["Category"].unique():
        category_recs = filtered[filtered["Category"] == category].sort_values("Suitability_Score", ascending=False).head(1)
        recommendations[category] = [
            {"Company": row["Company"], "Amount": invest_amount / len(filtered["Category"].unique()) if len(filtered["Category"].unique()) > 0 else invest_amount}
            for _, row in category_recs.iterrows()
        ]
    return recommendations

# PDF Generation
def generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_savings, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"WealthWise Plan for {name}", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "Powered by WealthWise | Built by xAI", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Financial Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Income: INR {income:,.2f}", ln=True)
    pdf.cell(0, 10, f"Predicted Savings: INR {predicted_savings:,.2f}", ln=True)
    pdf.cell(0, 10, f"Goal: {goal}", ln=True)
    pdf.cell(0, 10, f"Risk Tolerance: {risk_tolerance}", ln=True)
    pdf.cell(0, 10, f"Horizon: {horizon_years} years", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Investment Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    for category, recs in recommendations.items():
        if recs:
            pdf.cell(0, 10, f"{category}:", ln=True)
            for rec in recs:
                pdf.cell(0, 10, f"  - {rec['Company']}: INR {rec['Amount']:,.2f}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Budget Tips", ln=True)
    pdf.set_font("Arial", "", 10)
    for tip in tips:
        pdf.cell(0, 10, f"- {tip}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Peer Comparison", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Your Savings: INR {predicted_savings:,.2f} | Peer Avg: INR {peer_savings:,.2f}", ln=True)
    
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# Fetch Real-Time Stock Data
def get_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if "Time Series (5min)" not in data:
            return None, "Error: Invalid symbol, API key, or rate limit reached."
        time_series = data["Time Series (5min)"]
        df = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
        df.index = pd.to_datetime(df.index)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df, None
    except Exception as e:
        return None, f"Error: {str(e)}"

# Fetch Market News
def get_market_news(api_key, tickers="AAPL"):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={tickers}&apikey={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if "feed" not in data or not data["feed"]:
            return None, "No news available. Free key has limited access—try a premium key!"
        return data["feed"], None
    except Exception as e:
        return None, f"Error fetching news: {str(e)}"

# Main Application
def main():
    st.title("💰 Artha")
    st.markdown("Your ultimate wealth management companion! 🚀")

    # Initialize session state for tutorials
    if 'tutorial_steps' not in st.session_state:
        st.session_state.tutorial_steps = {
            "stock": [False] * 5,
            "investment": [False] * 5,
            "retirement": [False] * 5,
            "market": [False] * 5
        }

    # Load data
    stock_data = load_stock_data()
    survey_data = load_survey_data()
    financial_data = load_financial_data()

    # Train models
    stock_model, stock_r2 = None, 0.0
    if stock_data is not None:
        stock_model, stock_r2 = train_stock_model(stock_data)
    survey_model, survey_r2 = None, 0.0
    if survey_data is not None:
        survey_model, survey_r2 = train_survey_model(survey_data)
    retirement_model, retirement_r2 = None, 0.0
    if financial_data is not None:
        retirement_model, retirement_r2 = train_retirement_model(financial_data)
    investment_model = train_investment_model(investment_data)

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Insights")
        st.info("Explore your financial future!")
        if stock_data is not None:
            st.metric("Stock Model Accuracy (R²)", f"{stock_r2:.2f}")
        if survey_data is not None:
            st.metric("Savings Model Accuracy (R²)", f"{survey_r2:.2f}")
        if financial_data is not None:
            st.metric("Retirement Model Accuracy (R²)", f"{retirement_r2:.2f}")
        
        st.markdown("### 🔑 Your Market Data Pass")
        st.write("Paste your Alpha Vantage key for live data!")
        api_key = st.text_input("Paste Your Key Here", value="", type="password")
        st.markdown("[Get Free Key](https://www.alphavantage.co/)")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment", "🏡 Retirement Planning", "🌐 Live Market Insights"])

    with tab1:
        st.header("📈 Stock Market Adventure")
        st.markdown("Navigate NIFTY CONSUMPTION with precision! 🌟")

        # Interactive Tutorial
        if not all(st.session_state.tutorial_steps["stock"]):
            with st.expander("🎓 Learn How to Invest", expanded=True):
                st.markdown("Click each step!")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("📊 What’s NIFTY?", key="stock_step1"):
                        st.session_state.tutorial_steps["stock"][0] = True
                        st.write("Consumer companies index!")
                        st.audio("https://example.com/audio/nifty.mp3", format="audio/mp3")
                        st.balloons()
                with col2:
                    if st.button("⚙️ Set Inputs", key="stock_step2"):
                        st.session_state.tutorial_steps["stock"][1] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/stock_form.png", caption="Form")
                        st.audio("https://example.com/audio/inputs.mp3", format="audio/mp3")
                        st.balloons()
                with col3:
                    if st.button("🔮 Forecast", key="stock_step3"):
                        st.session_state.tutorial_steps["stock"][2] = True
                        st.metric("Sample", "₹5000", "₹200")
                        st.audio("https://example.com/audio/forecast.mp3", format="audio/mp3")
                        st.balloons()
                with col4:
                    if st.button("💡 Recommendations", key="stock_step4"):
                        st.session_state.tutorial_steps["stock"][3] = True
                        with st.expander("Sample"):
                            st.write("- Reliance: ₹2000")
                        st.audio("https://example.com/audio/recommendations.mp3", format="audio/mp3")
                        st.balloons()
                with col5:
                    if st.button("🔍 Chart", key="stock_step5"):
                        st.session_state.tutorial_steps["stock"][4] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/stock_chart.png", caption="Chart")
                        st.audio("https://example.com/audio/chart.mp3", format="audio/mp3")
                        st.balloons()

        with st.form(key="stock_form"):
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("⏳ Horizon (Months)", 1, 60, 12)
                invest_amount = st.number_input("💰 Amount (₹)", min_value=1000.0, value=6000.0, step=500.0)
            with col2:
                risk_tolerance = st.selectbox("🎲 Risk", ["Low", "Medium", "High"])
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
            submit = st.form_submit_button("🚀 Explore Market")
        
        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing..."):
                future = pd.DataFrame({"Day": [1], "Month": [horizon % 12 or 12], "Year": [2025 + horizon // 12]})
                predicted_price = stock_model.predict(future)[0]
                current_price = stock_data['close'].iloc[-1]
                growth = predicted_price - current_price
                horizon_years = horizon // 12 or 1
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goals)
            st.subheader("🔮 Market Forecast")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Price (₹)", f"₹{predicted_price:,.2f}", f"{growth:,.2f}")
            col2.metric("Growth Potential", f"{(growth/current_price)*100:.1f}%", "🚀" if growth > 0 else "📉")
            with st.expander("📊 Price Trend", expanded=True):
                fig = px.line(stock_data, x='Date', y='close', title="NIFTY Trend", hover_data=['open', 'high', 'low', 'volume'])
                fig.update_traces(line_color='#00ff00')
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("💡 Your Strategy")
            st.write(f"Goals: {', '.join(goals)}")
            progress = min(1.0, invest_amount / 100000)
            st.progress(progress)
            any_recommendations = False
            for category in ["Large Cap", "Medium Cap", "Low Cap", "Crypto"]:
                recs = recommendations.get(category, [])
                if recs:
                    any_recommendations = True
                    with st.expander(f"{category} Options"):
                        for rec in recs:
                            st.write(f"- **{rec['Company']}**: ₹{rec['Amount']:,.2f}")
            if not any_recommendations:
                st.info("No matches. Adjust inputs!")

    with tab2:
        st.header("🎯 Your Investment Journey")
        st.markdown("Craft a personalized plan! 🌈")

        # Interactive Tutorial
        if not all(st.session_state.tutorial_steps["investment"]):
            with st.expander("🎓 Learn How to Plan", expanded=True):
                st.markdown("Click each step!")
                progress = sum(st.session_state.tutorial_steps["investment"]) / 5
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("🏆 Why Plan?", key="invest_step1"):
                        st.session_state.tutorial_steps["investment"][0] = True
                        st.write("Tailored for YOU!")
                        st.audio("https://example.com/audio/why_plan.mp3", format="audio/mp3")
                        st.balloons()
                with col2:
                    if st.button("✏️ Add Details", key="invest_step2"):
                        st.session_state.tutorial_steps["investment"][1] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/invest_form.png", caption="Form")
                        st.audio("https://example.com/audio/details.mp3", format="audio/mp3")
                        st.balloons()
                with col3:
                    if st.button("🥧 Savings", key="invest_step3"):
                        st.session_state.tutorial_steps["investment"][2] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/pie_chart.png", caption="Pie")
                        st.audio("https://example.com/audio/savings.mp3", format="audio/mp3")
                        st.balloons()
                with col4:
                    if st.button("💰 Invest", key="invest_step4"):
                        st.session_state.tutorial_steps["investment"][3] = True
                        st.write("Sample: HDFC - ₹1000")
                        st.audio("https://example.com/audio/invest.mp3", format="audio/mp3")
                        st.balloons()
                with col5:
                    if st.button("📥 Download", key="invest_step5"):
                        st.session_state.tutorial_steps["investment"][4] = True
                        st.write("Get your PDF!")
                        st.audio("https://example.com/audio/download.mp3", format="audio/mp3")
                        st.balloons()
                st.progress(progress)

        with st.form(key="investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Name")
                income = st.number_input("💰 Income (₹)", min_value=0.0, step=1000.0)
                essentials = st.number_input("🍲 Essentials (₹)", min_value=0.0, step=100.0)
                non_essentials = st.number_input("🎉 Non-Essentials (₹)", min_value=0.0, step=100.0)
                debt_payment = st.number_input("💳 Debt (₹)", min_value=0.0, step=100.0)
            with col2:
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
                goal_amount = st.number_input("💎 Goal (₹)", min_value=0.0, step=1000.0, value=50000.0)
                risk_tolerance = st.selectbox("🎲 Risk", ["Low", "Medium", "High"])
                horizon_years = st.slider("⏳ Horizon (Years)", 1, 10, 3)
                invest_percent = st.slider("💸 % to Invest", 0, 100, 50)
            submit = st.form_submit_button("🚀 Get Plan")
        
        if submit and survey_data is not None and survey_model is not None:
            with st.spinner("Crafting plan..."):
                predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
                invest_amount = predicted_savings * (invest_percent / 100)
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goals)
                monthly_savings_needed = calculate_savings_goal(goal_amount, horizon_years)
                peer_avg_savings = survey_data["Savings"].mean()

            st.subheader("💰 Monthly Breakdown")
            breakdown_data = {"Essentials": essentials, "Non-Essentials": non_essentials, "Debt": debt_payment, "Savings": predicted_savings}
            fig = px.pie(values=list(breakdown_data.values()), names=list(breakdown_data.keys()), title="Spending vs. Savings")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💼 Investment Options")
            st.write(f"Goals: {', '.join(goals)} | Invest: ₹{invest_amount:,.2f}")
            for category in ["Large Cap", "Medium Cap", "Low Cap", "Crypto"]:
                recs = recommendations.get(category, [])
                if recs:
                    with st.expander(f"{category} Investments"):
                        for rec in recs:
                            st.write(f"- *{rec['Company']}*: ₹{rec['Amount']:,.2f}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Savings Progress")
                progress = min(1.0, predicted_savings / goal_amount) if goal_amount > 0 else 0
                st.progress(progress)
                st.write(f"₹{predicted_savings:,.2f} / ₹{goal_amount:,.2f}")
            with col2:
                st.subheader("📊 Peer Benchmark")
                st.bar_chart({"You": predicted_savings, "Peers": peer_avg_savings})

            st.subheader("⏰ Time to Goal")
            years_to_goal = (goal_amount / predicted_savings / 12) if predicted_savings > 0 else float('inf')
            timeline_data = pd.DataFrame({"Years": range(horizon_years + 1), "Savings": [predicted_savings * 12 * y for y in range(horizon_years + 1)]})
            fig = px.line(timeline_data, x="Years", y="Savings", title=f"Projected to ₹{goal_amount:,.2f}")
            fig.add_hline(y=goal_amount, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Time: {years_to_goal:.1f} years")

            with st.expander("💡 Budget Tips", expanded=True):
                tips = []
                median_non_essentials = survey_data["Non_Essentials"].median()
                if non_essentials > median_non_essentials:
                    tips.append(f"Cut non-essentials by ₹{non_essentials - median_non_essentials:,.2f}")
                if debt_payment > income * 0.3:
                    tips.append("Debt > 30% - refinance?")
                if predicted_savings < monthly_savings_needed:
                    tips.append(f"Save ₹{monthly_savings_needed - predicted_savings:,.2f} more/month")
                else:
                    tips.append("Great savings! Invest more?")
                for tip in tips:
                    st.write(f"- {tip}")

            pdf_buffer = generate_pdf(name, income, predicted_savings, ", ".join(goals), risk_tolerance, horizon_years, recommendations, peer_avg_savings, tips)
            st.download_button("📥 Download Plan", pdf_buffer, f"{name}_plan.pdf", "application/pdf")

    with tab3:
        st.header("🏡 Retirement Planning")
        st.markdown("Secure your golden years! 🌞")

        # Interactive Tutorial
        if not all(st.session_state.tutorial_steps["retirement"]):
            with st.expander("🎓 Learn How to Retire", expanded=True):
                st.markdown("Click through!")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("🌅 Why Now?", key="retire_step1"):
                        st.session_state.tutorial_steps["retirement"][0] = True
                        st.write("Plan early!")
                        st.audio("https://example.com/audio/why_retire.mp3", format="audio/mp3")
                        st.balloons()
                with col2:
                    if st.button("⏰ Age", key="retire_step2"):
                        st.session_state.tutorial_steps["retirement"][1] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/retire_form.png", caption="Form")
                        st.audio("https://example.com/audio/age.mp3", format="audio/mp3")
                        st.balloons()
                with col3:
                    if st.button("🎈 Inflation", key="retire_step3"):
                        st.session_state.tutorial_steps["retirement"][2] = True
                        st.write("₹1000 → ₹1300 in 10 yrs!")
                        st.audio("https://example.com/audio/inflation.mp3", format="audio/mp3")
                        st.balloons()
                with col4:
                    if st.button("🚀 Growth", key="retire_step4"):
                        st.session_state.tutorial_steps["retirement"][3] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/retire_graph.png", caption="Graph")
                        st.audio("https://example.com/audio/growth.mp3", format="audio/mp3")
                        st.balloons()
                with col5:
                    if st.button("💡 Tips", key="retire_step5"):
                        st.session_state.tutorial_steps["retirement"][4] = True
                        st.write("Save ₹500 more!")
                        st.audio("https://example.com/audio/tips.mp3", format="audio/mp3")
                        st.balloons()

        with st.form(key="retirement_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Income (₹)", min_value=0.0, step=1000.0)
                current_savings = st.number_input("🏦 Savings (₹)", min_value=0.0, step=1000.0)
            with col2:
                retirement_age = st.slider("👴 Retire Age", age + 1, 100, 65)
                monthly_expenses = st.number_input("💸 Expenses (₹)", min_value=0.0, step=500.0)
                inflation_rate = st.slider("📈 Inflation (%)", 0.0, 10.0, 3.0)
        
            st.subheader("Additional Income")
            income_sources = st.multiselect("Sources", ["Pension", "Rental Income", "Part-Time Work", "Other"])
            additional_income = sum([st.number_input(f"{source} (₹)", min_value=0.0, step=500.0, key=source) for source in income_sources])
        
            submit = st.form_submit_button("🚀 Plan Retirement")
    
        if submit and financial_data is not None and retirement_model is not None:
            with st.spinner("Projecting..."):
                years_to_retirement = retirement_age - age
                if years_to_retirement <= 0:
                    st.error("Retire age > current age!")
                else:
                    future_expenses = monthly_expenses * (1 + inflation_rate / 100) ** years_to_retirement if monthly_expenses > 0 else 0
                    retirement_goal = future_expenses * 12 * 20
                    annual_additional_income = additional_income * 12
                    retirement_goal = max(0, retirement_goal - annual_additional_income * 20)
                    predicted_savings = predict_retirement_savings(retirement_model, income, monthly_expenses)
                    retirement_wealth = forecast_retirement_savings(income, predicted_savings + current_savings, years_to_retirement)
        
                    st.subheader("🌟 Retirement Outlook")
                    col1, col2 = st.columns(2)
                    col1.metric("Wealth", f"₹{retirement_wealth:,.2f}")
                    col2.metric("Goal", f"₹{retirement_goal:,.2f}", f"{'Surplus' if retirement_wealth > retirement_goal else 'Shortfall'}: ₹{abs(retirement_wealth - retirement_goal):,.2f}")
                
                    st.subheader("📈 Trajectory")
                    trajectory = [forecast_retirement_savings(income, predicted_savings + current_savings, y) for y in range(years_to_retirement + 1)]
                    adjusted_goals = [max(0, future_expenses * 12 * min(y, 20) - (annual_additional_income * min(y, 20))) for y in range(years_to_retirement + 1)]
                    adjusted_goals = [float(x) if isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x)) else 0 for x in adjusted_goals]
                    x_values = list(range(years_to_retirement + 1))
                    if len(x_values) == len(trajectory) == len(adjusted_goals):
                        fig = px.line(x=x_values, y=trajectory, labels={"x": "Years", "y": "Wealth (₹)"}, title="Retirement Growth")
                        fig.add_scatter(x=x_values, y=adjusted_goals, mode='lines', name="Goal", line=dict(dash="dash", color="red"))
                        st.plotly_chart(fig, use_container_width=True)
                
                    st.subheader("💡 Tips")
                    if retirement_wealth < retirement_goal:
                        shortfall = (retirement_goal - retirement_wealth) / (years_to_retirement * 12)
                        st.write(f"- Save ₹{shortfall:,.2f} more/month")
                    if additional_income > 0:
                        st.write(f"- Extra ₹{additional_income:,.2f}/month helps!")
                    st.write(f"- Inflation: ₹{future_expenses:,.2f}/month")

    with tab4:
        st.header("🌐 Live Market Insights")
        st.markdown("Track portfolio and news live!")

        # Interactive Tutorial
        if not all(st.session_state.tutorial_steps["market"]):
            with st.expander("🎓 Learn How to Track", expanded=True):
                st.markdown("Unlock insights!")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    if st.button("🔑 Why Key?", key="market_step1"):
                        st.session_state.tutorial_steps["market"][0] = True
                        st.write("For live data!")
                        st.audio("https://example.com/audio/why_key.mp3", format="audio/mp3")
                        st.balloons()
                with col2:
                    if st.button("🔒 Get Key", key="market_step2"):
                        st.session_state.tutorial_steps["market"][1] = True
                        st.markdown("[Alpha Vantage](https://www.alphavantage.co/)")
                        st.audio("https://example.com/audio/get_key.mp3", format="audio/mp3")
                        st.balloons()
                with col3:
                    if st.button("📈 Stocks", key="market_step3"):
                        st.session_state.tutorial_steps["market"][2] = True
                        st.image("https://github.com/yourusername/artha-screenshots/raw/main/market_input.png", caption="Symbols")
                        st.audio("https://example.com/audio/stocks.mp3", format="audio/mp3")
                        st.balloons()
                with col4:
                    if st.button("📰 News", key="market_step4"):
                        st.session_state.tutorial_steps["market"][3] = True
                        st.write("Sample: 'Apple Up 5%'")
                        st.audio("https://example.com/audio/news.mp3", format="audio/mp3")
                        st.balloons()
                with col5:
                    if st.button("⚠️ Limits", key="market_step5"):
                        st.session_state.tutorial_steps["market"][4] = True
                        st.write("Free key: 5 calls/min")
                        st.audio("https://example.com/audio/limits.mp3", format="audio/mp3")
                        st.balloons()

        if not api_key:
            st.error("Add your Alpha Vantage key!")
        else:
            st.subheader("Live Portfolio")
            portfolio_input = st.text_area("Stock symbols (one per line):", "AAPL\nMSFT\nGOOGL\nTSLA")
            portfolio = [symbol.strip().upper() for symbol in portfolio_input.split("\n") if symbol.strip()]
            
            if st.button("Track Portfolio & News"):
                total_value = 0
                for symbol in portfolio:
                    with st.spinner(f"Fetching {symbol}..."):
                        df, error = get_stock_data(symbol, api_key)
                        if error or df is None:
                            st.error(f"{symbol}: {error}")
                            continue
                        latest_price = df["Close"].iloc[0]
                        previous_price = df["Close"].iloc[-1]
                        performance = ((latest_price - previous_price) / previous_price) * 100
                        total_value += latest_price
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{symbol} Price", f"${latest_price:.2f}", f"{performance:.2f}%")
                        with col2:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines"))
                            fig.update_layout(title=f"{symbol} Live Price")
                            st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Total Value: ${total_value:.2f}")

                st.subheader("Latest News")
                ticker_for_news = portfolio[0] if portfolio else "AAPL"
                with st.spinner(f"Fetching news for {ticker_for_news}..."):
                    news_feed, error = get_market_news(api_key, ticker_for_news)
                    if error or news_feed is None:
                        st.warning(error)
                    else:
                        for article in news_feed[:5]:
                            st.write(f"**{article['title']}**")
                            st.write(article["summary"])
                            st.write(f"[Read more]({article['url']})")

    st.markdown("---")
    st.write("Powered by WealthWise | Built by xAI")

if __name__ == "__main__":
    main()
