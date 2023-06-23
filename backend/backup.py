import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore tensorflow warnings. make sure this line is before importing tensorflow
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
seq_length = 30 # looking back 30 days and making predictions
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, seq_length)

# Step 4: Set up the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
    tf.keras.layers.Dropout(0.2), # dropout layers in the model architecture to randomly deactivate a fraction of neurons during training, reducing over-reliance on specific features.
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(90) # number of days we want to predict
])

############################################################################################################
model.save('stock_prediction_model.h5')
print("################################################Model saved successfully.")

# # Load the saved model
# loaded_model = tf.keras.models.load_model('stock_prediction_model.h5')
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# loaded_model.compile(optimizer=optimizer, loss='mean_squared_error')
# print("################################################Model loaded successfully.")

############################################################################################################

# Step 4: Train the model
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=0.001)  # Set the clipnorm parameter to prevent gradient explosion
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

# Step 5: Evaluate the model on test data
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Training loss: {train_loss:.6f}')
print(f'Test loss: {test_loss:.6f}')

# Step 6: Make predictions
last_sequence = X_test[-1:]
next_price = model.predict(last_sequence)
next_price = scaler.inverse_transform(next_price)
print(f'Next day prediction: {next_price[0][0]}\n')


# Step 7: Make predictions for the next 90 days
x_input = X_test[-seq_length:]  # Get the last seq_length data points from the test set
x_input_scaled = scaler.transform(x_input.reshape(-1, 1))  # Reshape and scale the input data
x_input_scaled = x_input_scaled[-seq_length:].reshape(1, seq_length, 1)
predictions_scaled = model.predict(x_input_scaled)  # Make predictions for the next 90 days
predictions = scaler.inverse_transform(predictions_scaled)  # Rescale the predictions to the original scale
last_date = features.iloc[-1]  # Get the last date in the test set
next_90_days = pd.date_range(last_date, periods=90, freq='D')  # Generate dates for the next 90 days
# Print the predicted stock prices for the next 90 days
for i in range(len(predictions)):
    print(f'{next_90_days[i].date()}: {predictions[i][0]}')

