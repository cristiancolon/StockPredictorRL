import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('aggr_data/all_data.csv')
actual_prices = data['close']
predicted_prices = data['Predictions_lstm']
sentiments = data['sentiment_score']

def calculate_position_size(capital, risk_per_trade, stop_loss_amount):
    return (capital * risk_per_trade) / stop_loss_amount

def calculate_modified_atr(close_prices, n=14):
    # Calculate the absolute differences between consecutive close prices
    close_diff = close_prices.diff().abs()
    # Compute a simple moving average of these differences
    atr = close_diff.rolling(window=n, min_periods=1).mean()
    return atr

def simple_trading_strategy(prices, predictions, sentiments, capital, risk_per_trade, indx):
    short_ma = prices[indx-25:indx].mean()
    long_ma = prices[indx-50:indx].mean()
    atr = calculate_modified_atr(prices).iloc[indx]
    last_price = prices[indx]
    predicted_future_price = predictions[indx]
    stop_loss_amount = 2 * atr

    if short_ma > long_ma and predicted_future_price > last_price and sentiments[indx] > 0:
        action = "Buy"
        amount = calculate_position_size(capital, risk_per_trade, stop_loss_amount)
        if capital >= amount:
            capital -= amount
            return action, amount, capital
    elif short_ma < long_ma and predicted_future_price < last_price and sentiments[indx] < 0:
        action = "Sell"
        amount = calculate_position_size(capital, risk_per_trade, stop_loss_amount)
        return action, amount, capital

    return "Hold", 0, capital


data['timestamp'] = pd.to_datetime(data['timestamp'])  # Assuming there's a Date column
# Example arrays/lists to hold transaction data, assume these get filled as you iterate through your strategy
buy_dates = []
sell_dates = []
profits = []
capital_timeline = []

capital = 10000
investment = 0
holdings = []

capital = 10000
risk_per_trade = 0.1  # 1% risk per trade
investment = 0
holdings = []  # List to track each purchase (amount and purchase price)

for i in range(50, data.shape[0]):
    action, amount, capital = simple_trading_strategy(actual_prices, predicted_prices, sentiments, capital, risk_per_trade, i)

    if action == 'Buy':
        investment += amount
        holdings.append((amount, actual_prices[i]))
        buy_dates.append(data.index[i])
    elif action == 'Sell' and investment > 0:
        total_value = sum(h[0] * (actual_prices[i] / h[1]) for h in holdings)
        profit = total_value - investment
        capital += total_value  # Include profit in the capital
        investment = 0
        holdings.clear()
        sell_dates.append(data.index[i])
        profits.append(profit)  # This should be calculated during the Sell
    capital_timeline.append(capital)

plt.figure(figsize=(14, 7))  # Set the size of the plot
plt.plot(data.index, data['close'], label='Close Price', color='lightgray')  # Plot close prices
plt.scatter(buy_dates, data.loc[buy_dates, 'close'], label='Buy', marker='^', color='green', alpha=0.7, s=100)
plt.scatter(sell_dates, data.loc[sell_dates, 'close'], label='Sell', marker='v', color='red', alpha=0.7, s=100)
cumulative_profit = np.cumsum(profits)  # Cumulative sum of profits
profit_dates = sell_dates  # Assuming profits are realized on sell dates

ax2 = plt.gca().twinx()  # Create a second y-axis
ax2.plot(profit_dates, cumulative_profit, label=f'Cumulative Profit={str(round(cumulative_profit[0], 2))}', color='blue', marker='o', linestyle='--')
ax2.set_ylabel('Cumulative Profit ($)')
plt.title('Stock Price and Transactions')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.grid(True)
plt.show()