import pandas as pd
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error

#feature engineering 

#separates the datetime value 

#takes into account seasons and allows model to recognize cyclic nature of climate
def cyclic_nature(df, col_name): 
    max_val = df[col_name].max(); 
    df[col_name + '_sin'] = np.sin(df[col_name] * (2 * np.pi / max_val))
    df[col_name + '_cos'] = np.cos(df[col_name] * (2 * np.pi / max_val))
    return df

#creates features representing values from previous time steps
def time_lags(df, col_name, lags):
    for lag in lags:
        df[f'{col_name}_lag_{lag}'] = df[col_name].shift(lag)
    return df.ffill()


# add the data to a dataframe
predict_df = pd.read_csv('predict_cleaned.csv')
actual_df = pd.read_csv('actual_cleaned.csv')

# feature engineering
predict_df = cyclic_nature(predict_df, 'hour')
predict_df = time_lags(predict_df, 'temp', [1, 2, 3])

# ensure there are no NaN values before scaling
predict_df = predict_df.ffill().bfill()

# select features for prediction
features = [col for col in predict_df.columns if col not in ['year', 'month', 'day', 'hour']]

# normalize that data 
scaler = StandardScaler()
scaled_pdf = scaler.fit_transform(predict_df)

# using the previous 30 days to predict that day
def create_sequences(data, seq_length): 
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : ( i + seq_length)]
        y = data[i + seq_length, -1]

        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


seq_length = 14 # each new hour is p
X, y = create_sequences(scaled_pdf, seq_length)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_model((seq_length, X.shape[2]))

# train the model and make predictions
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
last_sequence = scaled_pdf[-seq_length:]
predicted_solar_energy = []

for _ in range(len(actual_df) - len(predict_df)):
    last_sequence_reshaped = last_sequence.reshape(1, seq_length, X.shape[2])
    next_pred = model.predict(last_sequence_reshaped)[0][0]
    predicted_solar_energy.append(next_pred)
    last_sequence = np.roll(last_sequence, -1, axis=0)
    last_sequence[-1] = np.append(last_sequence[-1][:-1], next_pred)

# inverse transform the predictions
predicted_solar_energy = scaler.inverse_transform(np.column_stack((np.zeros((len(predicted_solar_energy), len(features)-1)), predicted_solar_energy)))[:, -1]

# get the actual solar energy values for the predicted period
actual_solar_energy = actual_df['solarenergy'].iloc[-len(predicted_solar_energy):].values

# calculate mse, mae and mape metrics
mae = mean_absolute_error(actual_solar_energy, predicted_solar_energy)
mse = mean_squared_error(actual_solar_energy, predicted_solar_energy)
mape = np.mean(np.abs((actual_solar_energy - predicted_solar_energy) / actual_solar_energy)) * 100

print("MAE:", mae)
print("MSE:", mse)
print("MAPE:", mape)

# save the predicted values in predicted df
predicted_df = pd.DataFrame({'predicted_solar_energy': predicted_solar_energy})
predicted_df.to_csv('predicted.csv', index=False)

# visualize results
plt.figure(figsize=(12, 6))
plt.plot(actual_solar_energy, label='Actual')
plt.plot(predicted_solar_energy, label='Predicted')
plt.title('Actual vs Predicted Solar Energy Output for the 4th Year')
plt.xlabel('Time')
plt.ylabel('Solar Energy Output')
plt.legend()
plt.show()

# Print NaN values
print("NaN values in predict_df:", predict_df.isna().sum())
print("NaN values in actual_df:", actual_df.isna().sum())