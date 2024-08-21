# Correcting the weather data generation by ensuring the number of columns matches the data
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Function to generate random weather data
def generate_weather_data(start_date, end_date):
    date_range = pd.date_range(start_date, end_date, freq='H')
    weather_data = []

    for date in date_range:
        month = date.month
        hour = date.hour
        
        # Define base temperatures and other metrics for each season
        if month in [12, 1, 2]:  # Winter
            base_temp = np.random.normal(0, 5)
            solar_radiation = np.random.uniform(0, 150)
            precip_chance = np.random.uniform(0, 0.3)
            snow_depth = max(0, np.random.normal(5, 5))
            condition = 'Snow' if precip_chance > 0.2 else 'Cloudy'
        elif month in [3, 4, 5]:  # Spring
            base_temp = np.random.normal(10, 5)
            solar_radiation = np.random.uniform(100, 300)
            precip_chance = np.random.uniform(0, 0.4)
            snow_depth = 0
            condition = 'Rain' if precip_chance > 0.3 else 'Partially cloudy'
        elif month in [6, 7, 8]:  # Summer
            base_temp = np.random.normal(25, 5)
            solar_radiation = np.random.uniform(200, 600)
            precip_chance = np.random.uniform(0, 0.2)
            snow_depth = 0
            condition = 'Clear' if precip_chance < 0.1 else 'Partially cloudy'
        else:  # Fall
            base_temp = np.random.normal(15, 5)
            solar_radiation = np.random.uniform(50, 250)
            precip_chance = np.random.uniform(0, 0.3)
            snow_depth = 0
            condition = 'Rain' if precip_chance > 0.2 else 'Cloudy'

        # Generate other weather parameters based on base temperature
        feels_like = base_temp + np.random.uniform(-2, 2)
        dew_point = base_temp - np.random.uniform(0, 5)
        humidity = np.random.uniform(50, 100)
        wind_speed = np.random.uniform(0, 15)
        wind_gust = wind_speed + np.random.uniform(0, 10)
        wind_dir = np.random.uniform(0, 360)
        pressure = np.random.uniform(1000, 1025)
        visibility = np.random.uniform(5, 10)
        solar_energy = solar_radiation * np.random.uniform(0.1, 1.0)
        uv_index = np.random.uniform(0, 10)
        cloud_cover = np.random.uniform(0, 100)

        weather_data.append([
            "Ann Arbor",
            date,
            round(base_temp, 1),
            round(feels_like, 1),
            round(dew_point, 1),
            round(humidity, 2),
            round(precip_chance, 2),
            np.nan,  # preciptype is not directly calculated
            condition,
            round(snow_depth, 1),
            round(snow_depth, 1),
            round(wind_gust, 1),
            round(wind_speed, 1),
            round(wind_dir, 1),
            round(pressure, 1),
            round(cloud_cover, 1),
            round(visibility, 1),
            round(solar_radiation, 1),
            round(solar_energy, 1),
            round(uv_index, 1),
            0,  # Severe risk
            condition,
            "clear-day" if condition == "Clear" else "cloudy",
            "KARB,KYIP,C4874,KDTW"
        ])
    
    # Convert to DataFrame
    columns = [
        "name", "datetime", "temp", "feelslike", "dew", "humidity", "precip", 
        "precipprob", "preciptype", "snow", "snowdepth", "windgust", "windspeed", 
        "winddir", "sealevelpressure", "cloudcover", "visibility", "solarradiation", 
        "solarenergy", "uvindex", "severerisk", "conditions", "icon", "stations"
    ]
    weather_df = pd.DataFrame(weather_data, columns=columns)
    
    return weather_df

# Generate data for 4 years
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)

weather_df = generate_weather_data(start_date, end_date)

#slice into 3 years
weather_df.to_csv('actual.csv')
weather_df.to_csv('predicted.csv')



 