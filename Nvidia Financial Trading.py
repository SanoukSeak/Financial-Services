
1. Data Preprocessing and Feature Engineering

!pip install tensorflow

!pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = pd.read_excel('nvidia_stock.xlsx')

# Clean column names
data.columns = data.columns.str.strip()

# Convert Date to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering: Calculate technical indicators (e.g., Moving Averages, RSI)
data['MA10'] = data['Adj Close'].rolling(window=10).mean()  # 10-day moving average
data['MA50'] = data['Adj Close'].rolling(window=50).mean()  # 50-day moving average
data['RSI'] = 100 - (100 / (1 + data['Adj Close'].pct_change(fill_method=None).apply(lambda x: np.max([x, 0])).rolling(window=14).mean() / 
    data['Adj Close'].pct_change(fill_method=None).apply(lambda x: np.abs(min(x, 0))).rolling(window=14).mean()))


# Drop NA values
data = data.dropna()

# Features and target
X = data[['Adj Close', 'Volume', 'MA10', 'MA50', 'RSI']]
y = data['Adj Close'].shift(-1)  # Predict next day's price
X = X[:-1]  # Align features and target
y = y.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

2. LSTM Model for Predicting Stock Prices

from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import numpy as np

# Example: Replace the input_shape argument with Input layer

# Sample data (replace with your own dataset)
X_train = np.random.rand(100, 10, 1)  # Example input shape (100 samples, 10 time steps, 1 feature)

# Sequential Model
model = Sequential()

# Define input layer with Input
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define the input shape here

# Add LSTM layer
model.add(LSTM(units=50, return_sequences=True))

# Add Dense layer for output
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()


3. XGBoost Model for Predicting Stock Prices

# XGBoost Model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
model_xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = model_xgb.predict(X_test)

# Evaluate with MSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}')



4. ARIMA Model for Predicting Stock Prices

Copy code
# ARIMA Model
train_data = data['Adj Close'][:int(0.8 * len(data))]
test_data = data['Adj Close'][int(0.8 * len(data)):]

# Fit ARIMA model
model_arima = ARIMA(train_data, order=(5, 1, 0))  # ARIMA(p,d,q)
model_arima_fit = model_arima.fit()

# Forecasting
forecast_arima = model_arima_fit.forecast(steps=len(test_data))

# Evaluate with MSE
mse_arima = mean_squared_error(test_data, forecast_arima)
print(f'ARIMA MSE: {mse_arima}')


5. Directional Accuracy Metric
This measures how often the model's direction (up or down) matches the actual stock price direction.

python
Copy code
# Directional Accuracy
def directional_accuracy(y_true, y_pred):
    y_true_direction = np.sign(np.diff(y_true))
    y_pred_direction = np.sign(np.diff(y_pred))
    accuracy = np.mean(y_true_direction == y_pred_direction)
    return accuracy

# Calculate directional accuracy for LSTM, XGBoost, and ARIMA
dir_acc_lstm = directional_accuracy(y_test[1:], y_pred_lstm.flatten())
dir_acc_xgb = directional_accuracy(y_test, y_pred_xgb)
dir_acc_arima = directional_accuracy(test_data, forecast_arima)

print(f'LSTM Directional Accuracy: {dir_acc_lstm}')
print(f'XGBoost Directional Accuracy: {dir_acc_xgb}')
print(f'ARIMA Directional Accuracy: {dir_acc_arima}')

6. Plot Predictions

# Plot LSTM predictions
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='True Prices')
plt.plot(y_test.index, y_pred_lstm, label='LSTM Predictions')
plt.title('LSTM Model Predictions')
plt.legend()
plt.show()

# Plot XGBoost predictions
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='True Prices')
plt.plot(y_test.index, y_pred_xgb, label='XGBoost Predictions')
plt.title('XGBoost Model Predictions')
plt.legend()
plt.show()

# Plot ARIMA predictions
plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data, label='True Prices')
plt.plot(test_data.index, forecast_arima, label='ARIMA Predictions')
plt.title('ARIMA Model Predictions')
plt.legend()
plt.show()

--------------------------------------------

## This is the code that examplifies the amplified fluctuations of both the dailiy and intraday strategies

import pandas as pd
import matplotlib.pyplot as plt

# Sample data for demonstration (replace with your data)
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
cum_daily_returns = pd.Series((1 + np.random.randn(100).cumsum() / 100), index=dates)
cum_intraday_returns = pd.Series((1 + np.random.randn(100).cumsum() / 100), index=dates)

# Normalize cumulative returns to start at 1 for better comparison
cum_daily_returns_normalized = cum_daily_returns / cum_daily_returns.iloc[0]
cum_intraday_returns_normalized = cum_intraday_returns / cum_intraday_returns.iloc[0]

# Amplify fluctuations by using a larger figure and customizing y-axis limits
plt.figure(figsize=(14, 7))
plt.plot(cum_daily_returns_normalized, label='Daily Strategy', linewidth=2, alpha=0.8)
plt.plot(cum_intraday_returns_normalized, label='Intraday Strategy', linewidth=2, alpha=0.8)

# Title and labels
plt.title('Amplified Cumulative Returns: Intraday vs Daily Strategy', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Normalized Cumulative Returns', fontsize=14)

# Adjust y-axis limits to amplify differences
plt.ylim([0.95, max(cum_intraday_returns_normalized.max(), cum_daily_returns_normalized.max()) * 1.05])

# Add grid, legend, and styling
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

--------------------------------------------------

## Examplified code with the LSTM:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import numpy as np

# Example: Replace the input_shape argument with Input layer

# Sample data (replace with your own dataset)
X_train = np.random.rand(100, 10, 1)  # Example input shape (100 samples, 10 time steps, 1 feature)

# Sequential Model
model = Sequential()

# Define input layer with Input
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define the input shape here

# Add LSTM layer
model.add(LSTM(units=50, return_sequences=True))

# Add Dense layer for output
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

-------------------------------------------------------


## Examplifed code that shows both the true and predicted values but some changes to the fill functions

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load NVIDIA stock data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Check for NaN values and handle them
if data.isnull().any().any():
    print("There are NaN values in the dataset. Filling NaNs...")
    data = data.fillna(method='ffill')  # Forward fill NaN values

# Use 'Close' prices for modeling
prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create sequences for LSTM
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(prices_scaled, time_steps)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sequential Model
model = Sequential()

# Define input layer with Input
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define the input shape here

# Add LSTM layer (simplified)
model.add(LSTM(units=50, return_sequences=False))  # Set return_sequences=False for a regression task

# Add Dense layer for output
model.add(Dense(1))

# Compile model with a lower learning rate
optimizer = Adam(learning_rate=0.001)  # Use a smaller learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Summary of the model
model.summary()

# Train the model (increase epochs if needed)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")

# Predict and reshape predictions for inverse transform
predictions = model.predict(X_test)

# Reshape predictions to 2D (ensure it's 2D for inverse transform)
predictions_reshaped = predictions.reshape(-1, 1)  # Flatten to 2D for inverse transformation

# Inverse transform the predictions and the actual values
predictions_rescaled = scaler.inverse_transform(predictions_reshaped)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label="True Prices", alpha=0.7)
plt.plot(predictions_rescaled, label="Predicted Prices", alpha=0.7)
plt.title("True vs Predicted Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()

----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load NVIDIA stock data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Check for NaN values and handle them
if data.isnull().any().any():
    print("There are NaN values in the dataset. Filling NaNs...")
    data = data.fillna(method='ffill')  # Forward fill NaN values

# Use 'Close' prices for modeling
prices = data['Close']

# Stationarity test - Augmented Dickey-Fuller test
def test_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print("The data is likely non-stationary (p > 0.05), differencing is required.")
    else:
        print("The data is likely stationary (p <= 0.05).")

# Test stationarity
test_stationarity(prices)

# If the data is not stationary, apply differencing
prices_diff = prices.diff().dropna()

# Test stationarity again after differencing
test_stationarity(prices_diff)

# Normalize the data (Optional)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices.values.reshape(-1, 1))

# Split the data into training and test sets (80% train, 20% test)
train_size = int(len(prices) * 0.8)
train, test = prices_scaled[:train_size], prices_scaled[train_size:]

# Fit an ARIMA model to the training data (tune p, d, q)
model = ARIMA(train, order=(5, 1, 0))  # Adjust order if needed after testing
model_fit = model.fit()

# Make predictions on the test set
forecast = model_fit.forecast(steps=len(test))

# Inverse transform the forecast and true values back to the original scale
forecast_rescaled = scaler.inverse_transform(forecast.reshape(-1, 1))
test_rescaled = scaler.inverse_transform(test.reshape(-1, 1))

# Evaluate the model using Mean Squared Error
mse_arima = mean_squared_error(test_rescaled, forecast_rescaled)
print(f'ARIMA MSE: {mse_arima}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(test_rescaled, label="True Prices", alpha=0.7)
plt.plot(forecast_rescaled, label="Predicted Prices (ARIMA)", alpha=0.7)
plt.title("True vs Predicted Stock Prices (ARIMA)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Stock Price", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

----------------------------------------------------------------

# Plot LSTM predictions
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='True Prices')
plt.plot(y_test.index, y_pred_lstm, label='LSTM Predictions')
plt.title('LSTM Model Predictions')
plt.legend()
plt.show()

# Plot XGBoost predictions
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='True Prices')
plt.plot(y_test.index, y_pred_xgb, label='XGBoost Predictions')
plt.title('XGBoost Model Predictions')
plt.legend()
plt.show()

# Plot ARIMA predictions
plt.figure(figsize=(12,6))
plt.plot(test_data.index, test_data, label='True Prices')
plt.plot(test_data.index, forecast_arima, label='ARIMA Predictions')
plt.title('ARIMA Model Predictions')
plt.legend()
plt.show()

----------------------------------------------------------------

## It plots the diff btw the Intraday and Daily strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Assuming 'Date' is the index or a column representing the date
data.set_index('Date', inplace=True)

# Convert 'Close' and 'Open' columns to numeric, forcing errors to NaN
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# Fill missing values using forward fill (without the deprecated method argument)
data['Close'] = data['Close'].ffill()  # Forward fill missing values
data['Open'] = data['Open'].ffill()  # Forward fill missing values

# Calculate daily strategy returns
data['Daily_Returns'] = data['Close'].pct_change()

# Calculate intraday strategy returns (e.g., open-to-close)
data['Intraday_Returns'] = (data['Close'] - data['Open']) / data['Open']

# Fill missing values in 'Intraday_Returns' if needed
data['Intraday_Returns'] = data['Intraday_Returns'].fillna(0)

# Calculate cumulative returns for each strategy
data['Daily_Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
data['Intraday_Cumulative_Returns'] = (1 + data['Intraday_Returns']).cumprod()

# Performance Metrics Function
def performance_metrics(returns):
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / volatility
    
    # Sortino Ratio (Assuming risk-free rate is 0 and downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output all metrics
    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Apply the performance metrics function to both strategies
daily_metrics = performance_metrics(data['Daily_Returns'])
intraday_metrics = performance_metrics(data['Intraday_Returns'])

# Print the results
print("Daily Strategy Performance Metrics:")
for key, value in daily_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIntraday Strategy Performance Metrics:")
for key, value in intraday_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Daily_Cumulative_Returns'], label='Daily Strategy')
plt.plot(data.index, data['Intraday_Cumulative_Returns'], label='Intraday Strategy')

# Add plot details
plt.title('Cumulative Returns: Intraday vs Daily Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

------------------------------------------\
# It shows the metrics btw the Daily and Intraday strategy

import pandas as pd
import numpy as np

# Load the data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Assuming 'Date' is the index or a column representing the date
data.set_index('Date', inplace=True)

# Convert 'Close' and 'Open' columns to numeric, forcing errors to NaN
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# Fill missing values using forward fill (without the deprecated method argument)
data['Close'] = data['Close'].ffill()  # Forward fill missing values
data['Open'] = data['Open'].ffill()  # Forward fill missing values

# Calculate daily strategy returns
data['Daily_Returns'] = data['Close'].pct_change()

# Calculate intraday strategy returns (e.g., open-to-close)
data['Intraday_Returns'] = (data['Close'] - data['Open']) / data['Open']

# Fill missing values in 'Intraday_Returns' if needed
data['Intraday_Returns'] = data['Intraday_Returns'].fillna(0)

# Performance Metrics Function
def performance_metrics(returns):
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / volatility
    
    # Sortino Ratio (Assuming risk-free rate is 0 and downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output all metrics
    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Apply the performance metrics function to both strategies
daily_metrics = performance_metrics(data['Daily_Returns'])
intraday_metrics = performance_metrics(data['Intraday_Returns'])

# Print the results
print("Daily Strategy Performance Metrics:")
for key, value in daily_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIntraday Strategy Performance Metrics:")
for key, value in intraday_metrics.items():
    print(f"{key}: {value:.4f}")


------------------------------------------

# This combines the metrics and the plot of the Daily and Intraday Strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Assuming 'Date' is the index or a column representing the date
data.set_index('Date', inplace=True)

# Convert 'Close' and 'Open' columns to numeric, forcing errors to NaN
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# Fill missing values using forward fill (without the deprecated method argument)
data['Close'] = data['Close'].ffill()  # Forward fill missing values
data['Open'] = data['Open'].ffill()  # Forward fill missing values

# Calculate daily strategy returns
data['Daily_Returns'] = data['Close'].pct_change()

# Calculate intraday strategy returns (e.g., open-to-close)
data['Intraday_Returns'] = (data['Close'] - data['Open']) / data['Open']

# Fill missing values in 'Intraday_Returns' if needed
data['Intraday_Returns'] = data['Intraday_Returns'].fillna(0)

# Calculate cumulative returns for each strategy
data['Daily_Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
data['Intraday_Cumulative_Returns'] = (1 + data['Intraday_Returns']).cumprod()

# Performance Metrics Function
def performance_metrics(returns):
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / volatility
    
    # Sortino Ratio (Assuming risk-free rate is 0 and downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output all metrics
    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Apply the performance metrics function to both strategies
daily_metrics = performance_metrics(data['Daily_Returns'])
intraday_metrics = performance_metrics(data['Intraday_Returns'])

# Print the results
print("Daily Strategy Performance Metrics:")
for key, value in daily_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIntraday Strategy Performance Metrics:")
for key, value in intraday_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Daily_Cumulative_Returns'], label='Daily Strategy')
plt.plot(data.index, data['Intraday_Cumulative_Returns'], label='Intraday Strategy')

# Add plot details
plt.title('Cumulative Returns: Intraday vs Daily Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

-----------------------------------------------

# Dumb Code: 
# Daily vs Intraday Strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Assuming 'Date' is the index or a column representing the date
data.set_index('Date', inplace=True)

# Convert 'Close' and 'Open' columns to numeric, forcing errors to NaN
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# Fill missing values using forward fill (without the deprecated method argument)
data['Close'] = data['Close'].ffill()  # Forward fill missing values
data['Open'] = data['Open'].ffill()  # Forward fill missing values

# Calculate daily strategy returns
data['Daily_Returns'] = data['Close'].pct_change()

# Calculate intraday strategy returns (e.g., open-to-close)
data['Intraday_Returns'] = (data['Close'] - data['Open']) / data['Open']

# Fill missing values in 'Intraday_Returns' if needed
data['Intraday_Returns'] = data['Intraday_Returns'].fillna(0)

# Calculate cumulative returns for each strategy
data['Daily_Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
data['Intraday_Cumulative_Returns'] = (1 + data['Intraday_Returns']).cumprod()

# Performance Metrics Function
def performance_metrics(returns):
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / volatility
    
    # Sortino Ratio (Assuming risk-free rate is 0 and downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output all metrics
    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Apply the performance metrics function to both strategies
daily_metrics = performance_metrics(data['Daily_Returns'])
intraday_metrics = performance_metrics(data['Intraday_Returns'])

# Print the results
print("Daily Strategy Performance Metrics:")
for key, value in daily_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIntraday Strategy Performance Metrics:")
for key, value in intraday_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Daily_Cumulative_Returns'], label='Daily Strategy')
plt.plot(data.index, data['Intraday_Cumulative_Returns'], label='Intraday Strategy')

# Add plot details
plt.title('Cumulative Returns: Intraday vs Daily Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

----------

# Whole timeline of Daily and Intraday Strategy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('nvidia_stock.xlsx', sheet_name='Sheet1')

# Clean column names
data.columns = data.columns.str.strip()

# Assuming 'Date' is the index or a column representing the date
data.set_index('Date', inplace=True)

# Convert 'Close' and 'Open' columns to numeric, forcing errors to NaN
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')

# Fill missing values using forward fill (without the deprecated method argument)
data['Close'] = data['Close'].ffill()  # Forward fill missing values
data['Open'] = data['Open'].ffill()  # Forward fill missing values

# Calculate daily strategy returns
data['Daily_Returns'] = data['Close'].pct_change()

# Calculate intraday strategy returns (e.g., open-to-close)
data['Intraday_Returns'] = (data['Close'] - data['Open']) / data['Open']

# Fill missing values in 'Intraday_Returns' if needed
data['Intraday_Returns'] = data['Intraday_Returns'].fillna(0)

# Calculate cumulative returns for each strategy
data['Daily_Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod()
data['Intraday_Cumulative_Returns'] = (1 + data['Intraday_Returns']).cumprod()

# Performance Metrics Function
def performance_metrics(returns):
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1  # Assuming 252 trading days in a year
    
    # Calculate volatility (standard deviation of returns)
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Sharpe Ratio (Assuming risk-free rate is 0)
    sharpe_ratio = annualized_return / volatility
    
    # Sortino Ratio (Assuming risk-free rate is 0 and downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else np.nan
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output all metrics
    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }

# Apply the performance metrics function to both strategies
daily_metrics = performance_metrics(data['Daily_Returns'])
intraday_metrics = performance_metrics(data['Intraday_Returns'])

# Print the results
print("Daily Strategy Performance Metrics:")
for key, value in daily_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nIntraday Strategy Performance Metrics:")
for key, value in intraday_metrics.items():
    print(f"{key}: {value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Daily_Cumulative_Returns'], label='Daily Strategy')
plt.plot(data.index, data['Intraday_Cumulative_Returns'], label='Intraday Strategy')

# Add plot details
plt.title('Cumulative Returns: Intraday vs Daily Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()


--------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

# Load data
data = pd.read_excel('nvidia_stock.xlsx')

# Clean column names
data.columns = data.columns.str.strip()

# Convert Date to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Feature Engineering: Calculate technical indicators (e.g., Moving Averages, RSI)
data['MA10'] = data['Adj Close'].rolling(window=10).mean()  # 10-day moving average
data['MA50'] = data['Adj Close'].rolling(window=50).mean()  # 50-day moving average
data['RSI'] = 100 - (100 / (1 + data['Adj Close'].pct_change(fill_method=None).apply(lambda x: np.max([x, 0])).rolling(window=14).mean() / 
    data['Adj Close'].pct_change(fill_method=None).apply(lambda x: np.abs(min(x, 0))).rolling(window=14).mean()))


# Drop NA values
data = data.dropna()

# Features and target
X = data[['Adj Close', 'Volume', 'MA10', 'MA50', 'RSI']]
y = data['Adj Close'].shift(-1)  # Predict next day's price
X = X[:-1]  # Align features and target
y = y.dropna()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

--------------------------------------------------

# XGBoost Model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
model_xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = model_xgb.predict(X_test)

# Evaluate with MSE
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}')

-------------------------------------------------

model_xgb.fit(X_train, y_train)

---------------------------------------------------

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80-20 split, no shuffling for time series data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

-----------------------------------------------------

# This used to be the right code of XGBOOST Model

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Train the XGBoost model
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
model_xgb.fit(X_train, y_train)

# Predictions
y_pred_xgb = model_xgb.predict(X_test)

# Evaluate with Mean Squared Error (MSE)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost MSE: {mse_xgb}')

# Plotting True vs Predicted Values
plt.figure(figsize=(14, 7))

# Plot true values (y_test)
plt.plot(y_test, label='True Prices', color='blue', linestyle='-', marker='o', markersize=4, alpha=0.8)

# Plot predicted values (y_pred_xgb)
plt.plot(y_pred_xgb, label='Predicted Prices (XGBoost)', color='red', linestyle='--', marker='x', markersize=5, alpha=0.8)

# Adding title and axis labels
plt.title("True vs Predicted Stock Prices (XGBoost)", fontsize=18, weight='bold')
plt.xlabel("Time (Days)", fontsize=14)
plt.ylabel("Stock Price (USD)", fontsize=14)

# Adding Mean Squared Error to the title
plt.title(f"True vs Predicted Stock Prices (XGBoost)\nMSE: {mse_xgb:.4f}", fontsize=18, weight='bold')

# Adding legend
plt.legend(loc='upper left', fontsize=12)

# Adding grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show the plot
plt.tight_layout()  # Ensure everything fits well in the plot
plt.show()

--------------------------------------------

# It shows the vertical lines of dividend dates only wo the volatility

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel('nvidia_stock.xlsx')

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Coerce invalid dates to NaT
data.set_index('Date', inplace=True)

# Identify second occurrences of duplicated dates
duplicated_dates = data.index[data.index.duplicated(keep=False)]  # All duplicate occurrences
second_duplicates = duplicated_dates[1::2].unique()  # Select every second occurrence

print(f"Duplicated second dates (Earnings Announcements): {second_duplicates}")

# Plot vertical lines for second duplicate dates
plt.figure(figsize=(10, 6))

# Highlight second duplicate dates with red dotted lines
for event in second_duplicates:
    plt.axvline(x=event, color='red', linestyle='--')

plt.title('Earnings Announcement Dates')
plt.xlabel('Date')
plt.ylabel('Indicator for Duplicated Second Dates')
plt.tight_layout()
plt.show()

------------------------------------------------

# This code shows the disconnected earning announcement and volatility trend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Remove any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Clean data (drop rows with missing Adjusted Close)
data = data.dropna(subset=['Adj Close'])

# Calculate daily returns
data['Returns'] = data['Adj Close'].pct_change()

# Define earnings announcement dates or event dates
events = pd.to_datetime([
    '2023-02-22', '2023-05-23', '2023-08-18', '2023-11-15'  # Example dates
])

# Calculate rolling volatility (standard deviation of returns over 30 days)
data['Volatility'] = data['Returns'].rolling(window=30).std()

# Plot volatility and highlight event dates
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Volatility'], label='Rolling Volatility (30 days)', color='blue')

# Highlight event dates
for event in events:
    plt.axvline(x=event, color='red', linestyle='--', label='Earnings Announcement' if event == events[0] else "")

plt.title('NVIDIA Stock Volatility and Earnings Announcements')
plt.xlabel('Date')
plt.ylabel('Volatility (30-day rolling)')
plt.legend()
plt.tight_layout()
plt.show()

------------------------------------------------

## Adjusted price of Nvida overtime

# Step 2: Inspect the data
print("First 5 rows of the data:")
print(data.head())

print("\nDataset Information:")
data.info()

# Step 3: Clean the data
# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Check for duplicates and remove them
data = data.drop_duplicates()

# Step 4: Format the data
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data.set_index('Date', inplace=True)  # Set 'Date' as the index

# Ensure 'Adj Close' column exists
if 'Adj Close' not in data.columns:
    raise ValueError("The file must contain a column named 'Adj Close'.")

# Step 5: Exploratory Data Analysis
# Plot the Adjusted Close Price
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Adj Close'], color='blue')
plt.title('NVIDIA Stock - Adjusted Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.grid(True)
plt.show()

# Step 6: Calculate and Validate Daily Returns
data['Returns'] = data['Adj Close'].pct_change()

# Display descriptive statistics
print("\nDescriptive Statistics for Adjusted Close and Returns:")
print(data[['Adj Close', 'Returns']].describe())

---------------------------------------

## This code is prompted to test the affects of dividend announcement over volatility behavior using ANOVA, but there are limited dividend dates to proceed


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
data = pd.read_excel('nvidia_stock.xlsx')

# Clean column names
data.columns = data.columns.str.strip()

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Coerce invalid dates to NaT

# Identify dividend dates (already identified)
dividend_dates = second_duplicates['Date']  # assuming you already have dividend dates

# Drop rows where 'Close' column has missing values
data = data.dropna(subset=['Close'])

# Calculate daily returns (fill_method=None to avoid deprecated warning)
data['Returns'] = data['Close'].pct_change(fill_method=None)

# Define event window (e.g., 5 days before and after dividend announcement)
event_window = 5  # You can adjust this window size

# Create a function to calculate volatility for a given window
def calculate_volatility(data, window=5):
    return data['Returns'].rolling(window).std()

# Create periods around dividend announcements: pre-event, event, post-event
volatility_data = []
for dividend_date in dividend_dates:
    # Define the window before and after the dividend date
    pre_event = data[(data['Date'] >= dividend_date - pd.Timedelta(days=event_window)) & 
                     (data['Date'] < dividend_date)]
    post_event = data[(data['Date'] > dividend_date) & 
                      (data['Date'] <= dividend_date + pd.Timedelta(days=event_window))]
    
    # Check if there are enough data points for each period
    if len(pre_event) >= event_window and len(post_event) >= event_window:
        pre_volatility = calculate_volatility(pre_event, window=5).iloc[-1]  # volatility of last day of pre-event window
        post_volatility = calculate_volatility(post_event, window=5).iloc[0]  # volatility of first day of post-event window
        volatility_data.append({'Dividend Date': dividend_date, 'Pre-event Volatility': pre_volatility, 'Post-event Volatility': post_volatility})
    else:
        print(f"Not enough data for dividend date: {dividend_date}")

# Convert volatility data to DataFrame
volatility_df = pd.DataFrame(volatility_data)

# Check if we have enough data for ANOVA
if len(volatility_df) > 1:
    # Perform ANOVA to test if there's a significant difference between pre-event and post-event volatility
    f_stat, p_value = stats.f_oneway(volatility_df['Pre-event Volatility'].dropna(), 
                                      volatility_df['Post-event Volatility'].dropna())

    # Display the results of ANOVA
    print("ANOVA Results:")
    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")

    # If p-value < 0.05, we can reject the null hypothesis that there's no significant difference
    if p_value < 0.05:
        print("There is a significant difference in volatility before and after dividend announcements.")
    else:
        print("There is no significant difference in volatility before and after dividend announcements.")

    # Optionally, visualize the volatility
    volatility_df[['Pre-event Volatility', 'Post-event Volatility']].plot(kind='box', vert=False)
    plt.title('Volatility Before and After Dividend Announcement')
    plt.xlabel('Volatility')
    plt.show()
else:
    print("Not enough data for ANOVA test.")
