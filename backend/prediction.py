import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore tensorflow warnings. make surer this line is before importting tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Step 1: Load data from CSV
# Read the CSV file containing the stock data from yahoo finance for ONEX
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_file = os.path.join(parent_directory, 'data/uploads', 'ONEX_stock_Price_close.csv')
data = pd.read_csv(data_file)
data = data.dropna()  # Remove rows with NaN values
features = data['Date']
target = data['Close']

# Convert date string to datetime format
features = pd.to_datetime(features)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, shuffle=False)

# Step 3: Scale the variables using Min-Max scaling
scaler = MinMaxScaler()
scaler.fit(X_train.values.reshape(-1, 1))

X_train_scaled = scaler.transform(X_train.values.reshape(-1, 1))
X_test_scaled = scaler.transform(X_test.values.reshape(-1, 1))

scaler.fit(y_train.values.reshape(-1, 1))

y_train_scaled = scaler.transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

# Create sequences for training data
def create_sequences(x_data, y_data, seq_length):
    X = []
    y = []
    for i in range(len(x_data) - seq_length):
        X.append(x_data[i:i+seq_length])
        y.append(y_data[i+seq_length])
    return np.array(X), np.array(y)

# Convert the scaled data back to sequences
seq_length = 30
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, seq_length)


# Step 4: Set up the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(1)
])
 
# Step 4: Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1)
# optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Set the clipnorm parameter to prevent gradient explosion

# Step 5: Evaluate the model on test data
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training loss: {train_loss:.6f}')
print(f'Test loss: {test_loss:.6f}')

# Step 6: Make predictions
last_sequence = X_test[-90:]
next_price = model.predict(last_sequence)
next_price = scaler.inverse_transform(next_price)
print(f'Next day prediction: {next_price[0][0]}\nNext Month prediction: {next_price[29][0]}\nNext quarter prediction: {next_price[89][0]}\n')
# return next_price[0][0], next_price[0][0], next_price[0][0]
