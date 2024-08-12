import pandas as pd
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dataset = pd.read_csv('data5.csv', index_col=0, parse_dates=True)
dataset.reset_index(inplace=True)
columns_to_plot = ['feelslike', 'dew', 'precip', 'cloudcover', 'humidity', 'solarradiation', 'windspeed']
dataset.drop(['name','preciptype', 'conditions', 'icon', 'stations', 'snow', 'snowdepth', 'severerisk'], axis=1, inplace=True)
dataset.dropna(inplace=True)
dataset.set_index('datetime', inplace=True)
dataset.to_csv('data5_cleaned.csv', index=False)

dataset[columns_to_plot].plot()
plt.xticks(rotation=45)
plt.show()