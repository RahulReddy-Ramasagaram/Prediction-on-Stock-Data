

# Part 1: Import Libraries and Download Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

data = pd.read_csv("dataset_zhou.csv")

TARGET = 'Low' # Open, High, Low, CLose
data = data[['Date', 'Open', 'High', 'Low', 'Close']]
data.head()

columns = ['Open', 'High', 'Low', 'Close']
correlation_matrix = data[columns].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[TARGET].values.reshape(-1, 1))
    return scaled_data, scaler

# Preprocess the data
scaled_data, scaler = preprocess_data(data)

# Define the sequence length
seq_length = 10
# Function to create sequences of data
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Split the data into training and testing sets for each model
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - seq_length:]

# Split the data into X_train and y_train for LSTM
X_train_lstm, y_train_lstm = create_sequences(train_data, seq_length), train_data[seq_length:]
X_test_lstm, y_test_lstm = create_sequences(test_data, seq_length), test_data
y_test_lstm = y_test_lstm[seq_length:]

X_test_lstm.shape, y_test_lstm.shape

# Split the data into X_train and y_train for CNN
X_train_cnn, y_train_cnn = create_sequences(train_data, seq_length), train_data[seq_length:]
X_test_cnn, y_test_cnn = create_sequences(test_data, seq_length), test_data
y_test_cnn = y_test_cnn[seq_length:]

X_test_cnn.shape, y_test_cnn.shape

# Split the data into X_train and y_train for Conv1D-LSTM
X_train_conv1d_lstm, y_train_conv1d_lstm = create_sequences(train_data, seq_length), train_data[seq_length:]
X_test_conv1d_lstm, y_test_conv1d_lstm = create_sequences(test_data, seq_length), test_data
y_test_conv1d_lstm = y_test_conv1d_lstm[seq_length:]

X_test_conv1d_lstm.shape, y_test_conv1d_lstm.shape

# Function to build the LSTM model
def build_lstm_model(seq_length):
    model = Sequential()
    model.add(LSTM(50, input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the LSTM model
lstm_model = build_lstm_model(seq_length)
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16)

# Function to build the CNN model
def build_cnn_model(seq_length):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, 1)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the CNN model
cnn_model = build_cnn_model(seq_length)
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=16)

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Predict
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_cnn = cnn_model.predict(X_test_cnn)

lstm_loss = lstm_model.evaluate(X_test_lstm, y_test_lstm)
cnn_loss = cnn_model.evaluate(X_test_cnn, y_test_cnn)

# Calculate evaluation metrics for LSTM
lstm_mse = mean_squared_error(y_test_lstm, y_pred_lstm)
lstm_mae = mean_absolute_error(y_test_lstm, y_pred_lstm)
lstm_rmse = np.sqrt(lstm_mse)
lstm_r2 = r2_score(y_test_lstm, y_pred_lstm)
lstm_mape = mean_absolute_percentage_error(y_test_lstm, y_pred_lstm)

print(f"LSTM Model Evaluation Metrics ({TARGET}):")
print(f"LSTM Mean Squared Error (MSE): {lstm_mse:.4f}")
print(f"LSTM Mean Absolute Error (MAE): {lstm_mae:.4f}")
print(f"LSTM Root Mean Squared Error (RMSE): {lstm_rmse:.4f}")
print(f"LSTM R^2 Score: {lstm_r2:.4f}")
print(f"LSTM Mean Absolute Percentage Error (MAPE): {lstm_mape:.4f}")

# Calculate evaluation metrics for CNN
cnn_mse = mean_squared_error(y_test_cnn, y_pred_cnn)
cnn_mae = mean_absolute_error(y_test_cnn, y_pred_cnn)
cnn_rmse = np.sqrt(cnn_mse)
cnn_r2 = r2_score(y_test_cnn, y_pred_cnn)
cnn_mape = mean_absolute_percentage_error(y_test_cnn, y_pred_cnn)

print(f"CNN Model Evaluation Metrics ({TARGET}):")
print(f"CNN Mean Squared Error (MSE): {cnn_mse:.4f}")
print(f"CNN Mean Absolute Error (MAE): {cnn_mae:.4f}")
print(f"CNN Root Mean Squared Error (RMSE): {cnn_rmse:.4f}")
print(f"CNN R^2 Score: {cnn_r2:.4f}")
print(f"CNN Mean Absolute Percentage Error (MAPE): {cnn_mape:.4f}")

# Save LSTM model architecture and weights
lstm_model.save("lstm_model.h5")

# Save CNN model architecture and weights
cnn_model.save("cnn_model.h5")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Create the figure and plot
plt.figure(figsize=(18, 8))
plt.plot(y_test_lstm, label='Real Values (LSTM)', color='blue', linestyle='--')
plt.plot(y_pred_lstm, label='Predictions (LSTM)', color='red')
plt.plot(y_test_cnn, label='Real Values (CNN)', color='green', linestyle='--')
plt.plot(y_pred_cnn, label='Predictions (CNN)', color='orange')


# Grid
plt.grid(True, linestyle='--', alpha=0.6)

# Title with performance metrics
plt.title(f'Stock Price Prediction vs. Reality')

# Axis labels and legend
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Create a function for plotting
def plot_results(y_test, y_pred, model_name):

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real Values', color='blue')
    plt.plot(y_pred, label='Predictions', color='red')

    # Grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Title with performance metrics
    plt.title(f'Stock Price Prediction vs. Reality ({model_name})')

    # Axis labels and legend
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    # Show the plot
    plt.show()

# Plot results for LSTM
plot_results(y_test_lstm, y_pred_lstm, 'LSTM')

# Plot results for CNN
plot_results(y_test_cnn, y_pred_cnn, 'CNN')