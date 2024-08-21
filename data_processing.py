import pandas as pd

def extract_time(df):
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    
    # After extracting, reorder the columns as needed
    df = df[['year', 'month', 'day', 'hour'] + [col for col in df.columns if col not in ['year', 'month', 'day', 'hour', 'time']]]
    return df

predict_df = pd.read_csv('3_year_climate_data.csv', parse_dates=['time'])
actual_df = pd.read_csv('4_year_actual_values.csv', parse_dates=['time'])
predict_df = extract_time(predict_df)
actual_df = extract_time(actual_df)

# Drop the 'precipprob' column
predict_df = predict_df.drop(columns=["dew", "humidity", "preciptype", "snow", "snowdepth", "sealevelpressure", "visibility", "severerisk", "conditions", "icon", "stations" ])
actual_df = actual_df.drop(["name", "dew", "humidity", "preciptype", "snow", "snowdepth", "sealevelpressure", "visibility", "severerisk", "uvindex", "conditions", "icon", "stations"], axis = 1, inplace = True)
predict_df.dropna()
actual_df.dropna()
columns_to_add = ['datetime', 'solarenergy']
actual_df = actual_df[columns_to_add]; 
actual_df.rename(columns={"datetime" : "time"})

# print the head to verify 
print('sample data head: ')
print(predict_df.head())
print('actual data head: ')
print(actual_df.head())

actual_df.to_csv('actual_data.csv', index = False)
predict_df.to_csv('sample_data.csv', index = False)