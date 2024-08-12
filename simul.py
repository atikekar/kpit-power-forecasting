import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop

# Load the dataset
dataset1 = pd.read_csv('data1_cleaned.csv')
dataset2 = pd.read_csv('data2_cleaned.csv')
dataset3 = pd.read_csv('data3_cleaned.csv')
dataset4 = pd.read_csv('data4_cleaned.csv')
dataset5 = pd.read_csv('data5_cleaned.csv')

dataset = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])

# Normalization function
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

# Data import function
def import_data(train_dataframe, dev_dataframe, test_dataframe): 
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')

    max_test = np.max(dataset[:, 14])
    min_test = np.min(dataset[:, 14])
    scale_factor = max_test - min_test 

    # Initialize data_min and data_max lists
    data_min = np.zeros(dataset.shape[1])
    data_max = np.zeros(dataset.shape[1])
    
    # Create training set 
    for i in range(0, 15): 
        data_min[i] = np.amin(dataset[:, i], axis=0)
        data_max[i] = np.amax(dataset[:, i], axis=0)
        dataset[:, i] = normalize_data(dataset[:, i], data_min[i], data_max[i])
    
    train_data = dataset[:, 0:12]
    train_labels = dataset[:, 12:14]

    # Create dev set
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], data_min[i], data_max[i])

    dev_data = dataset[:, 0:12]
    dev_labels = dataset[:, 12:14]

    # Create test set
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')
    
    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], data_min[i], data_max[i])

    test_data = dataset[:, 0:12]
    test_labels = dataset[:, 12:14]

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor

# Function to build the neural network model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2)  # Output layer with 2 neurons (solar energy and wind energy)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    return model

# Example usage
# Split your dataset into train, validation (dev), and test sets
train_df, temp_df = train_test_split(dataset, test_size=0.3, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor = import_data(train_df, dev_df, test_df)

# Build the model
input_shape = train_data.shape[1]
model = build_model(input_shape)

# Train the model
history = model.fit(train_data, train_labels, epochs=15, validation_data=(dev_data, dev_labels))

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