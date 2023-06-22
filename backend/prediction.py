import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

# Scale the target variable using Min-Max scaling
scaler = MinMaxScaler()
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

# Normalize the features using Min-Max scaling
features_scaled = scaler.fit_transform(features.values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data) * 0.9)  # 90% for training, 10% for testing
X_train = features_scaled[:train_size]
y_train = target_scaled[:train_size]
X_test = features_scaled[train_size:]
y_test = target_scaled[train_size:]

# Create sequences for training data
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Number of previous days' prices to consider for prediction
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Step 2: Set up the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])
 
# Step 3: Train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1)
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Set the clipnorm parameter to prevent gradient explosion
epochs = 5
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(X_train)
        loss = tf.keras.losses.mean_squared_error(y_train, predictions)
    
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)  # Clip the gradients
    
    # Update weights
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
    
    # Evaluate the updated loss
    updated_predictions = model(X_train)
    updated_loss = tf.keras.losses.mean_squared_error(y_train, updated_predictions)
    
    # Print the loss
    print(f'Epoch {epoch+1}/{epochs}, Loss: {updated_loss.numpy().mean():.6f}')  # Convert loss to scalar


# Step 4: Evaluate the model on test data
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training loss: {train_loss:.6f}')
print(f'Test loss: {test_loss:.6f}')

# Step 5: Make predictions
last_sequence = train_data[-seq_length:]  # Last sequence from training data
prediction_days = 1  # Predictions for the next day
future_price_sequence = []

for _ in range(prediction_days):
    last_sequence = last_sequence.reshape(1, seq_length, 1)
    next_price = model.predict(last_sequence)
    future_price_sequence.append(next_price[0])
    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_price.reshape(1, 1, 1)), axis=1)

# Inverse scale the predicted prices
future_price_sequence = scaler.inverse_transform(future_price_sequence)

print(f'Next day prediction: {future_price_sequence[0][0]}')

# Predictions for the next month (assuming 30 trading days)
prediction_days = 30
last_sequence = train_data[-seq_length:]
future_price_sequence = []

for _ in range(prediction_days):
    last_sequence = last_sequence.reshape(1, seq_length, 1)
    next_price = model.predict(last_sequence)
    future_price_sequence.append(next_price[0])
    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_price.reshape(1, 1, 1)), axis=1)

future_price_sequence = scaler.inverse_transform(future_price_sequence)
next_month_average = np.mean(future_price_sequence)
print(f'Next month prediction: {next_month_average}')

# Predictions for the next quarter (assuming 90 trading days)
prediction_days = 90
last_sequence = train_data[-seq_length:]
future_price_sequence = []

for _ in range(prediction_days):
    last_sequence = last_sequence.reshape(1, seq_length, 1)
    next_price = model.predict(last_sequence)
    future_price_sequence.append(next_price[0])
    last_sequence = np.concatenate((last_sequence[:, 1:, :], next_price.reshape(1, 1, 1)), axis=1)

future_price_sequence = scaler.inverse_transform(future_price_sequence)
next_quarter_average = np.mean(future_price_sequence)
print(f'Next quarter prediction: {next_quarter_average}')