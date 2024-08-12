import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
dataset1 = pd.read_csv('data1_cleaned.csv')
dataset2 = pd.read_csv('data2_cleaned.csv')
dataset3 = pd.read_csv('data3_cleaned.csv')
dataset4 = pd.read_csv('data4_cleaned.csv')
dataset5 = pd.read_csv('data5_cleaned.csv')

dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

# Split the data into features and target
X = scaled_data[:, :-1]  # All columns except the last one
y = scaled_data[:, -1]   # The last column (solar energy)

# Split the dataset into train, validation (dev), and test sets
train_data, temp_data, train_labels, temp_labels = train_test_split(X, y, test_size=0.3, random_state=42)
dev_data, test_data, dev_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

# Function to build the ANN model
def build_ann_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for solar energy prediction
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    return model

# Build the ANN model
input_shape = train_data.shape[1]
model = build_ann_model(input_shape)

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

history = model.fit(train_data, train_labels, 
                    epochs=100, 
                    validation_data=(dev_data, dev_labels),
                    callbacks=[early_stop, reduce_lr])

# Evaluate the model on test data
test_loss, test_mae, test_mse = model.evaluate(test_data, test_labels)

# Print evaluation results
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test MSE: {test_mse}")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Make predictions
predictions = model.predict(test_data)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test_labels, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted Solar Energy')
plt.xlabel('Time')
plt.ylabel('Solar Energy')
plt.legend()
plt.show()