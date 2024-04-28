import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm

# Title of the Streamlit app
st.title("Crypto ETF Portfolio Simulation")

# Input to get the desired year
year = st.number_input("Enter Year", min_value=2010, max_value=2023, value=2020, step=1)

# Define the start and end dates for the given year
start_date = dt.datetime(year, 1, 1)
end_date = dt.datetime(year, 12, 31)

# Define top cryptocurrencies for the given year
cryptos = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BCH-USD', 'ADA-USD', 'LTC-USD', 'EOS-USD', 'BNB-USD', 
           'XTZ-USD', 'XLM-USD', 'LINK-USD', 'TRX-USD', 'NEO-USD', 'IOTA-USD', 'DASH-USD']

# Fetch historical data
@st.cache  # Cache the data for performance
def fetch_crypto_data(cryptos, start, end):
    crypto_data = {}
    for crypto in cryptos:
        data = yf.download(crypto, start=start, end=end)
        crypto_data[crypto] = data['Adj Close']
    return crypto_data

crypto_data = fetch_crypto_data(cryptos, start_date, end_date)

# Function to calculate market cap-based weights
def calculate_weights(crypto_data, date):
    adj_closes = {k: v.loc[date] for k, v in crypto_data.items() if date in v.index}
    total_market_cap = sum(adj_closes.values())
    return {k: v / total_market_cap for k, v in adj_closes.items()}

# Rebalance monthly based on market cap
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Start of each month
portfolio_values = pd.DataFrame(index=crypto_data['BTC-USD'].index)

for date in monthly_dates:
    weights = calculate_weights(crypto_data, date)
    for crypto, weight in weights.items():
        portfolio_values[crypto] = crypto_data[crypto] * weight

portfolio_values['Total'] = portfolio_values.sum(axis=1)

# Plot the portfolio value over time
st.write("Portfolio Value over the Year")
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values['Total'])
plt.title(f"Simulated Portfolio Value ({year})")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
st.pyplot(plt)

# Calculate risk metrics
# Define a risk-free rate (e.g., 2% annual rate)
risk_free_rate = 0.02


portfolio_returns = portfolio_values['Total'].pct_change().dropna()
std_dev = portfolio_returns.std()  # Standard deviation
annualized_return = portfolio_returns.mean() * 252  # Annualized return
annualized_std_dev = std_dev * np.sqrt(252)  # Annualized standard deviation
risk_free_rate = 0.02  # 2% annual risk-free rate
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std_dev

# Display risk metrics
st.write("Risk Metrics")
st.write(f"Standard Deviation: {std_dev:.4f}")
st.write(f"Sharpe Ratio: {sharpe_ratio:.4f}")
