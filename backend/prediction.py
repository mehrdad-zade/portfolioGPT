import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # ignore tensorflow warnings. make sure this line is before importing tensorflow
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import timedelta


##################################################################################### Step 1: Load data from CSV

# Read the CSV file containing the stock data from yahoo finance for ONEX
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_file = os.path.join(parent_directory, 'data/uploads', 'ONEX_stock_Price_close.csv')
data = pd.read_csv(data_file)
data = data.dropna()  # Remove rows with NaN values
date_feature = data['Date']
prices = data['Close']
date_series = pd.to_datetime(date_feature) # Convert date string to datetime format

##################################################################################### Step 2: Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(date_series, prices, test_size=0.1, shuffle=False) # don't shuffle time series data. X = date_series, y = prices

##################################################################################### Step 3: Scale the variables using Min-Max scaling

scaler = MinMaxScaler()
# Fit and transform the training data
scaled_prices_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
# Transform the testing data
scaled_prices_test = scaler.transform(y_test.values.reshape(-1, 1))

##################################################################################### Step 3: set parameters

window_size = 30  # number of days to look back
number_of_prediction_days = 90
drop_out_rate = 0.2
learning_rate = 0.001
loss_algorithm = 'mean_squared_error'
batch_size = 32
epochs = 30
number_of_neurons = 64

##################################################################################### step 4: Create sequences

def create_sequences(X, scaled_prices_train):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size - number_of_prediction_days):
        X_seq.append(scaled_prices_train[i:i+window_size])
        y_seq.append(scaled_prices_train[i+window_size : i+window_size+number_of_prediction_days])
    return np.array(X_seq), np.array(y_seq)

# Creating sequences for training & testing data
X_train_seq, y_train_seq = create_sequences(X_train, scaled_prices_train)
X_test_seq, y_test_seq = create_sequences(X_test, scaled_prices_test)

def trainNN():
    ##################################################################################### Step 5: Set up the neural network architecture

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(number_of_neurons, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        tf.keras.layers.LSTM(number_of_neurons, return_sequences=True),
        tf.keras.layers.Dropout(drop_out_rate), # dropout layers in the model architecture to randomly deactivate a fraction of neurons during training, reducing over-reliance on specific features.
        tf.keras.layers.LSTM(number_of_neurons, return_sequences=True),
        tf.keras.layers.LSTM(number_of_neurons, return_sequences=False),
        tf.keras.layers.Dense(number_of_neurons, activation='relu'),
        tf.keras.layers.Dropout(drop_out_rate),
        tf.keras.layers.Dense(number_of_prediction_days) 
    ])

    ##################################################################################### Step 6: Train the model

    optimizer = tf.keras.optimizers.Adam(clipnorm=1.0, learning_rate=learning_rate)  # Set the clipnorm parameter to prevent gradient explosion
    model.compile(optimizer=optimizer, loss=loss_algorithm)
    history = model.fit(X_train_seq, y_train_seq, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save('stock_prediction_model.h5')
    print("--->Model saved successfully.")

    # # Load the saved model
    # loaded_model = tf.keras.models.load_model('stock_prediction_model.h5')
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # loaded_model.compile(optimizer=optimizer, loss='mean_squared_error')
    # print("--->Model loaded successfully.")

    ##################################################################################### Step 7: Evaluate the model on test data

    train_loss = model.evaluate(X_train_seq, y_train_seq, verbose=0)
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    print(f'Training loss: {train_loss:.6f}')
    print(f'Test loss: {test_loss:.6f}')

    ##################################################################################### Step 8: Make predictions

def getPred(model):
    # Reshape the test data to match the input shape of the LSTM model
    X_test = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2]))

    # Retrieve the last sequence from the test data and reshape it to match the input shape of the LSTM model
    last_sequence = X_test[-1].reshape((1, X_test_seq.shape[1], X_test_seq.shape[2]))

    # Make a prediction for the next day
    prediction = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction).tolist()
    # Print the predicted value for the next day
    print(f"Predicted value for the next day: {prediction[0][0]}")
    next_90_days_predictions = prediction[0]
    return next_90_days_predictions


##################################################################################### step 9: Plot the predictions

def getPredictionPlot():
    loaded_model = tf.keras.models.load_model('stock_prediction_model.h5') # Load the saved model
    next_90_days_predictions = getPred(loaded_model)
    # stock historical price data
    df_Date = data
    df_Date['Date'] = pd.to_datetime(df_Date['Date'])

    # Get the last date in the DataFrame and increment by number_of_prediction_days. then add the new dates along with next_90_days_predictions to a new df
    last_date = df_Date['Date'].iloc[-1]
    new_dates = [last_date + timedelta(days=i) for i in range(1, len(next_90_days_predictions)+1)]
    new_data = pd.DataFrame({'Date': new_dates, 'Close': next_90_days_predictions})

    plt.switch_backend('Agg') # so that your server does not try to create (and then destroy) GUI windows that will never be seen.
    plt.figure(figsize = (15,10))
    plt.plot(data['Date'], data['Close'], label='OneX - Daily Stock Price', color='blue')
    plt.plot(new_data['Date'], new_data['Close'], label='OneX - Predicted Price', color='red')
    plt.legend(loc='best')
    
    plt.savefig('static/tmp_prediction.png')
    
    # plt.show()


# trainNN()
# getPredictionPlot()