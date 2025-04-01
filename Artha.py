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

# --- Data Loading Functions ---
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
            st.warning("⚠️ 'projected_savings' not found in CSV. Using 20% of income as a placeholder.")
        else:
            df = df.rename(columns={col_map["projected_savings"]: text"Projected_Savings"})
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

# --- Model Training Functions ---
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

# --- Predictive and Utility Functions ---
def predict_savings(model, income, essentials, non_essentials, debt_payment):
    input_df = pd.DataFrame({
        "Income": [income],
        "Essentials": [essentials],
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
        (
            investment_data["Goal_Encoded"].isin(goal_encoded_list) | 
            (investment_data["Goal_Encoded"] == goal_map["No specific goal"])
        )
    ]
    
    recommendations = {}
    for category in filtered["Category"].unique():
        category_recs = filtered[filtered["Category"] == category].sort_values("Suitability_Score", ascending=False).head(1)
        recommendations[category] = [
            {"Company": row["Company"], "Amount": invest_amount / len(filtered["Category"].unique()) if len(filtered["Category"].unique()) > 0 else invest_amount}
            for _, row in category_recs.iterrows()
        ]
    return recommendations

# --- PDF Generation ---
def generate_pdf(name, income, predicted_savings, goal, risk_tolerance, horizon_years, recommendations, peer_savings, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"WealthWise Investment Plan for {name}", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "Powered by WealthWise | Built with love by xAI", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Financial Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Income: INR {income:,.2f}", ln=True)
    pdf.cell(0, 10, f"Predicted Savings: INR {predicted_savings:,.2f}", ln=True)
    pdf.cell(0, 10, f"Goal: {goal}", ln=True)
    pdf.cell(0, 10, f"Risk Tolerance: {risk_tolerance}", ln=True)
    pdf.cell(0, 10, f"Investment Horizon: {horizon_years} years", ln=True)
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
    pdf.cell(0, 10, f"Your Savings: INR {predicted_savings:,.2f} | Peer Average: INR {peer_savings:,.2f}", ln=True)
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# --- API Functions ---
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

# --- Chatbot Class ---
class Chatbot:
    def __init__(self):
        self.responses = {
            "hi": "Hello! I’m your Artha guide. How can I assist you today?",
            "what can you do": "I’m here to help you navigate Artha! Click the 'Chatbot: Explain This Tab' button in any tab to see a visual guide, or ask me anything!",
            "start tour": "Let’s explore Artha! Click the 'Chatbot: Explain This Tab' button in any tab to see a visual guide.",
            "thanks": "You’re welcome! Anything else I can help with?",
            "bye": "See you later! Enjoy mastering your finances with Artha!"
        }
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = ["👋 Hi! I’m your Artha Chatbot. Click 'Chatbot: Explain This Tab' in any tab to see a visual guide, or ask me anything!"]

    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        if user_input in self.responses:
            return self.responses[user_input]
        elif "tab" in user_input:
            return "Click the 'Chatbot: Explain This Tab' button in the tab you want to learn about to see a visual guide!"
        elif "how" in user_input:
            return "I can guide you visually! Click the 'Chatbot: Explain This Tab' button in the tab you want to learn about."
        else:
            return "Hmm, I’m not sure about that. Try clicking 'Chatbot: Explain This Tab' in any tab to see a visual guide!"

# --- Flowchart Display Function ---
def display_flowchart(tab_name):
    flowcharts = {
        "Stock Investments": ["📅", "💰", "🎲", "🎯", "🚀"],  # Horizon, Amount, Risk, Goals, Submit
        "Personalized Investment": ["👤", "💰", "🍲", "🎉", "💳", "🎯", "💎", "🎲", "⏳", "💸", "🚀"],  # Name, Income, Essentials, Non-Essentials, Debt, Goals, Goal Amount, Risk, Horizon, Invest %, Submit
        "Retirement Planning": ["🎂", "💰", "🏦", "👴", "💸", "📈", "💼", "🚀"],  # Age, Income, Savings, Retirement Age, Expenses, Inflation, Income Sources, Submit
        "Live Market Insights": ["🔑", "📝", "🚀"]  # API Key, Stock Symbols, Submit
    }
    
    steps = flowcharts.get(tab_name, [])
    if not steps:
        return
    
    flowchart_html = """
    <style>
        .flowchart-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            padding: 20px;
            background-color: #1e1e1e;
            border-radius: 10px;
            margin: 10px 0;
        }
        .flowchart-step {
            font-size: 40px;
            transition: transform 0.3s;
        }
        .flowchart-step:hover {
            transform: scale(1.2);
        }
        .arrow {
            font-size: 30px;
            color: #ff4d4d;
        }
    </style>
    <div class="flowchart-container">
    """
    
    for i, step in enumerate(steps):
        flowchart_html += f'<span class="flowchart-step">{step}</span>'
        if i < len(steps) - 1:
            flowchart_html += '<span class="arrow">➡️</span>'
    
    flowchart_html += """
    </div>
    <script>
        const steps = document.querySelectorAll('.flowchart-step');
        steps.forEach(step => {
            step.addEventListener('mouseover', () => {
                const audio = new Audio('https://www.soundjay.com/buttons/pop.mp3');
                audio.play();
            });
        });
    </script>
    """
    
    st.markdown(flowchart_html, unsafe_allow_html=True)

# --- Main Application ---
def main():
    st.title("💰 Artha")
    st.markdown("Your ultimate wealth management companion! 🚀")

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
        st.info("Explore your financial future with these tools!")
        if stock_data is not None:
            st.metric("Stock Model Accuracy (R²)", f"{stock_r2:.2f}")
        if survey_data is not None:
            st.metric("Savings Model Accuracy (R²)", f"{survey_r2:.2f}")
        if financial_data is not None:
            st.metric("Retirement Model Accuracy (R²)", f"{retirement_r2:.2f}")
        
        st.markdown("### 🔑 Your Market Data Pass")
        st.write("To see live stock prices and news, we need a 'key'—think of it like a ticket to unlock real-time market updates! It’s free and easy to get.")
        api_key = st.text_input("Paste Your Key Here", value="", type="password", 
                               help="This is a special code from Alpha Vantage that lets us fetch live stock data just for you!")
        st.markdown("""
        **Why do I need this?**  
        It’s your VIP pass to see what’s happening in the stock market right now—like checking the latest price of Apple or Tesla!
        
        **How to Get It:**  
        1. Visit [Alpha Vantage](https://www.alphavantage.co/).  
        2. Click 'Get Free API Key' and sign up with your email.  
        3. Copy the code they give you (e.g., 'X7K9P2M4Q1').  
        4. Paste it here and start tracking!
        """)

        # Chatbot Integration in Sidebar
        st.markdown("---")
        st.subheader("💬 Chat with Your Artha Guide")
        chatbot = Chatbot()
        for message in st.session_state.chat_history:
            st.write(message)
        user_input = st.text_input("Ask me anything!", key="chat_input")
        if st.button("Send", key="chat_send"):
            if user_input:
                st.session_state.chat_history.append(f"**You**: {user_input}")
                response = chatbot.get_response(user_input)
                st.session_state.chat_history.append(f"**Chatbot**: {response}")
                st.rerun()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Investments", "🎯 Personalized Investment", "🏡 Retirement Planning", "🌐 Live Market Insights"])

    # --- Tab 1: Stock Investments ---
    with tab1:
        if st.button("Chatbot: Explain This Tab", key="tab1_help"):
            display_flowchart("Stock Investments")
            st.session_state.chat_history.append("**Chatbot**: Here’s a visual guide to the Stock Investments tab!")

        st.header("📈 Stock Market Adventure")
        st.markdown("Navigate the NIFTY CONSUMPTION index with precision! 🌟")
        
        with st.form(key="stock_form"):
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("⏳ Investment Horizon (Months)", 1, 60, 12)
                invest_amount = st.number_input("💰 Amount to Invest (₹)", min_value=1000.0, value=6000.0, step=500.0)
            with col2:
                risk_tolerance = st.selectbox("🎲 Risk Appetite", ["Low", "Medium", "High"])
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
            submit = st.form_submit_button("🚀 Explore Market")
        
        if submit and stock_data is not None and stock_model is not None:
            with st.spinner("Analyzing your investment strategy..."):
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
                fig = px.line(stock_data, x='Date', y='close', title="NIFTY CONSUMPTION Trend", 
                             hover_data=['open', 'high', 'low', 'volume'])
                fig.update_traces(line_color='#00ff00')
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("💡 Your Investment Strategy")
            st.write(f"Goals Selected: {', '.join(goals)}")
            progress = min(1.0, invest_amount / 100000)
            st.progress(progress)
            any_recommendations = False
            for category in ["Large Cap", "Medium Cap", "Low Cap", "Crypto"]:
                recs = recommendations.get(category, [])
                if recs:
                    any_recommendations = True
                    with st.expander(f"{category} Options"):
                        for rec in recs:
                            st.write(f"- **{rec['Company']}**: Invest ₹{rec['Amount']:,.2f}")
            if not any_recommendations:
                st.info("No investment options match your criteria. Try increasing your investment amount or adjusting your risk tolerance/goals.")

    # --- Tab 2: Personalized Investment ---
    with tab2:
        if st.button("Chatbot: Explain This Tab", key="tab2_help"):
            display_flowchart("Personalized Investment")
            st.session_state.chat_history.append("**Chatbot**: Here’s a visual guide to the Personalized Investment tab!")

        st.header("🎯 Your Investment Journey")
        st.markdown("Craft a personalized plan for wealth growth! 🌈")
        with st.form(key="investment_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("👤 Your Name")
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                essentials = st.number_input("🍲 Essentials (₹)", min_value=0.0, step=100.0)
                non_essentials = st.number_input("🎉 Non-Essentials (₹)", min_value=0.0, step=100.0)
                debt_payment = st.number_input("💳 Debt Payment (₹)", min_value=0.0, step=100.0)
            with col2:
                goals = st.multiselect("🎯 Goals", ["Wealth growth", "Emergency fund", "Future expenses", "No specific goal"], default=["Wealth growth"])
                goal_amount = st.number_input("💎 Total Goal Amount (₹)", min_value=0.0, step=1000.0, value=50000.0)
                risk_tolerance = st.selectbox("🎲 Risk Tolerance", ["Low", "Medium", "High"])
                horizon_years = st.slider("⏳ Horizon (Years)", 1, 10, 3)
                invest_percent = st.slider("💸 % of Savings to Invest", 0, 100, 50)
            submit = st.form_submit_button("🚀 Get Your Plan")
        
        if submit and survey_data is not None and survey_model is not None:
            with st.spinner("Crafting your personalized plan..."):
                predicted_savings = predict_savings(survey_model, income, essentials, non_essentials, debt_payment)
                invest_amount = predicted_savings * (invest_percent / 100)
                recommendations = predict_investment_strategy(investment_model, invest_amount, risk_tolerance, horizon_years, goals)
                monthly_savings_needed = calculate_savings_goal(goal_amount, horizon_years)
                peer_avg_savings = survey_data["Savings"].mean()

            st.subheader("💰 Your Monthly Breakdown")
            breakdown_data = {
                "Essentials": essentials,
                "Non-Essentials": non_essentials,
                "Debt Payment": debt_payment,
                "Savings": predicted_savings
            }
            fig = px.pie(values=list(breakdown_data.values()), names=list(breakdown_data.keys()), title="Spending vs. Savings")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💼 Your Investment Options")
            st.write(f"Goals Selected: {', '.join(goals)}")
            st.write(f"Amount to Invest: ₹{invest_amount:,.2f} ({invest_percent}% of ₹{predicted_savings:,.2f})")
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
            months_to_goal = goal_amount / predicted_savings if predicted_savings > 0 else float('inf')
            years_to_goal = months_to_goal / 12
            timeline_data = pd.DataFrame({
                "Years": range(horizon_years + 1),
                "Savings": [predicted_savings * 12 * y for y in range(horizon_years + 1)]
            })
            fig = px.line(timeline_data, x="Years", y="Savings", title=f"Projected Savings to Reach ₹{goal_amount:,.2f}")
            fig.add_hline(y=goal_amount, line_dash="dash", line_color="red", annotation_text="Goal")
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Estimated Time to Goal: {years_to_goal:.1f} years at current savings rate")

            with st.expander("💡 Personalized Budget Tips", expanded=True):
                tips = []
                median_non_essentials = survey_data["Non_Essentials"].median()
                if non_essentials > median_non_essentials:
                    tips.append(f"Reduce non-essentials by ₹{non_essentials - median_non_essentials:,.2f} (peer median: ₹{median_non_essentials:,.2f}).")
                if debt_payment > income * 0.3:
                    tips.append("Debt payment exceeds 30% of income - consider refinancing or cutting expenses.")
                if predicted_savings < monthly_savings_needed:
                    shortfall = monthly_savings_needed - predicted_savings
                    tips.append(f"Boost savings by ₹{shortfall:,.2f}/month to meet your goal in {horizon_years} years.")
                else:
                    tips.append("Great job! Your savings exceed your goal - consider increasing your investment percentage.")
                if "Wealth growth" in goals and risk_tolerance == "Low":
                    tips.append(f"For wealth growth, consider medium-risk options to boost returns over {horizon_years} years.")
                for tip in tips:
                    st.write(f"- {tip}")

            st.subheader("🎲 Risk Tolerance Assessment")
            risk_map = {"Low": "Safe", "Medium": "Balanced", "High": "Aggressive"}
            st.write(f"Your Profile: *{risk_map[risk_tolerance]}*")
            if risk_tolerance == "Low" and horizon_years > 5:
                st.info("Long horizon with low risk? You could explore medium-risk options for better returns.")
            elif risk_tolerance == "High" and horizon_years < 3:
                st.warning("Short horizon with high risk? Consider safer options to protect your funds.")

            pdf_buffer = generate_pdf(name, income, predicted_savings, ", ".join(goals), risk_tolerance, horizon_years, recommendations, peer_avg_savings, tips)
            st.download_button("📥 Download Your Plan", pdf_buffer, f"{name}_investment_plan.pdf", "application/pdf")

    # --- Tab 3: Retirement Planning ---
    with tab3:
        if st.button("Chatbot: Explain This Tab", key="tab3_help"):
            display_flowchart("Retirement Planning")
            st.session_state.chat_history.append("**Chatbot**: Here’s a visual guide to the Retirement Planning tab!")

        st.header("🏡 Retirement Planning")
        st.markdown("Secure your golden years with smart savings! 🌞")
    
        with st.form(key="retirement_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Current Age", min_value=18, max_value=100, value=30)
                income = st.number_input("💰 Monthly Income (₹)", min_value=0.0, step=1000.0)
                current_savings = st.number_input("🏦 Current Savings (₹)", min_value=0.0, step=1000.0)
            with col2:
                retirement_age = st.slider("👴 Retirement Age", age + 1, 100, 65)
                monthly_expenses = st.number_input("💸 Expected Monthly Expenses (₹)", min_value=0.0, step=500.0)
                inflation_rate = st.slider("📈 Expected Inflation Rate (%)", 0.0, 10.0, 3.0)
        
            st.subheader("Additional Income Sources in Retirement")
            income_sources = st.multiselect("Select Sources", ["Pension", "Rental Income", "Part-Time Work", "Other"])
            additional_income = 0
            for source in income_sources:
                amount = st.number_input(f"Monthly {source} (₹)", min_value=0.0, step=500.0, key=source)
                additional_income += amount
        
            submit = st.form_submit_button("🚀 Plan My Retirement")
    
        if submit and financial_data is not None and retirement_model is not None:
            with st.spinner("Projecting your retirement..."):
                years_to_retirement = retirement_age - age
                if years_to_retirement <= 0:
                    st.error("🚨 Retirement age must be greater than current age!")
                else:
                    future_expenses = monthly_expenses * (1 + inflation_rate / 100) ** years_to_retirement if monthly_expenses > 0 else 0
                    retirement_goal = future_expenses * 12 * 20
                    annual_additional_income = additional_income * 12
                    retirement_goal -= annual_additional_income * 20
                    retirement_goal = max(0, retirement_goal)
                    predicted_savings = predict_retirement_savings(retirement_model, income, monthly_expenses)
                    retirement_wealth = forecast_retirement_savings(income, predicted_savings + current_savings, years_to_retirement)
        
                    st.subheader("🌟 Retirement Outlook")
                    col1, col2 = st.columns(2)
                    col1.metric("Projected Wealth", f"₹{retirement_wealth:,.2f}")
                    col2.metric("Inflation-Adjusted Goal (After Income)", f"₹{retirement_goal:,.2f}",
                                f"{'Surplus' if retirement_wealth > retirement_goal else 'Shortfall'}: ₹{abs(retirement_wealth - retirement_goal):,.2f}")
                
                    st.subheader("📈 Savings Trajectory")
                    trajectory = [forecast_retirement_savings(income, predicted_savings + current_savings, y) for y in range(years_to_retirement + 1)]
                    adjusted_goals = [max(0, future_expenses * 12 * min(y, 20) - (annual_additional_income * min(y, 20))) for y in range(years_to_retirement + 1)]
                    adjusted_goals = [float(x) if isinstance(x, (int, float)) and not (np.isnan(x) or np.isinf(x)) else 0 for x in adjusted_goals]
                    x_values = list(range(years_to_retirement + 1))
                    if len(x_values) != len(trajectory) or len(x_values) != len(adjusted_goals):
                        st.error("Data length mismatch detected. Unable to plot trajectory.")
                    else:
                        fig = px.line(x=x_values, y=trajectory, labels={"x": "Years", "y": "Wealth (₹)"}, title="Retirement Growth vs Inflation-Adjusted Goal")
                        fig.add_scatter(x=x_values, y=adjusted_goals, mode='lines', name="Adjusted Goal", line=dict(dash="dash", color="red"))
                        st.plotly_chart(fig, use_container_width=True)
                
                    st.subheader("💡 Retirement Tips")
                    if retirement_wealth < retirement_goal:
                        shortfall = (retirement_goal - retirement_wealth) / (years_to_retirement * 12)
                        st.write(f"- Increase monthly savings by ₹{shortfall:,.2f} to meet your inflation-adjusted goal.")
                    if additional_income > 0:
                        st.write(f"- Your additional income of ₹{additional_income:,.2f}/month reduces your savings burden significantly!")
                    st.write(f"- Inflation at {inflation_rate}% increases your future expenses to ₹{future_expenses:,.2f}/month.")
                    st.write("- Consider adjusting investments for higher returns if needed.")

    # --- Tab 4: Live Market Insights ---
    with tab4:
        if st.button("Chatbot: Explain This Tab", key="tab4_help"):
            display_flowchart("Live Market Insights")
            st.session_state.chat_history.append("**Chatbot**: Here’s a visual guide to the Live Market Insights tab!")

        st.header("🌐 Live Market Insights")
        st.markdown("Track your portfolio and stay updated with market news—your key unlocks this magic!")

        with st.expander("How to Use This?"):
            st.write("""
            1. **Add Your Key**: Paste your Alpha Vantage key in the sidebar (see instructions there!).
            2. **Pick Stocks**: Edit the list below or use these popular ones:
               - AAPL (Apple)
               - MSFT (Microsoft)
               - GOOGL (Google)
               - TSLA (Tesla)
            3. **Track & Read**: Click 'Track Portfolio & News' to see live prices and headlines!
            """)
            st.info("No key yet? Follow the sidebar steps—it’s free and takes just a minute!")

        if not api_key:
            st.error("Oops! Please add your Alpha Vantage key in the sidebar to access live market insights.")
        else:
            st.subheader("Live Portfolio Tracking")
            portfolio_input = st.text_area("Enter stock symbols (one per line):", "AAPL\nMSFT\nGOOGL\nTSLA")
            portfolio = [symbol.strip().upper() for symbol in portfolio_input.split("\n") if symbol.strip()]
            
            if st.button("Track Portfolio & News"):
                total_value = 0
                for symbol in portfolio:
                    with st.spinner(f"Fetching live data for {symbol}..."):
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
                            st.metric(label=f"{symbol} Current Price", value=f"${latest_price:.2f}", delta=f"{performance:.2f}%", delta_color="normal")
                        with col2:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{symbol} Price"))
                            fig.update_layout(title=f"{symbol} Live Price (Last 100 intervals)", xaxis_title="Time", yaxis_title="Price (USD)")
                            st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Total Portfolio Value: ${total_value:.2f}")

                st.subheader("Latest Market News")
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
                st.info("News access is limited with a free Alpha Vantage key. For more, consider a premium key.")

    st.markdown("---")
    st.write("Powered by WealthWise | Built with love by xAI")

if __name__ == "__main__":
    main()
