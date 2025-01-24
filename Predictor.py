import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def get_stock_data(ticker, start, end):
    """
    Description: Download stock data
    """
    data = yf.download(ticker, start=start, end=end)
    return data['Close']

def prepare_data(data, window_size):
    """
    Description: Data preparation
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_train, y_train = [], []
    for i in range(window_size, len(scaled_data)):
        x_train.append(scaled_data[i-window_size:i, 0])
        y_train.append(scaled_data[i, 0])
    
    return np.array(x_train), np.array(y_train), scaler

def build_lstm_model(input_shape):
    """
    Description: Long Short-Term Memory Model
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = "2023-01-01"
    window_size = 60
    
    # Load Data
    data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare Data
    x, y, scaler = prepare_data(data, window_size)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    # Split into training and testing sets
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build Model
    model = build_lstm_model((x_train.shape[1], 1))
    
    # Train Model
    model.fit(x_train, y_train, batch_size=32, epochs=20)
    
    # Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_scaled, label="True Prices")
    plt.plot(predictions, label="Predicted Prices")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
