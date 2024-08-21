With increasing funding in renewable energy sources (RESs), wind, photovoltaic (PV) and thermal energy are contributing to a larger portion of energy grid utilities. 
While environmentally friendly, RES output can fluctuate dramatically based on time of day, weather patterns and seasons. This ultimately creates a gap in energy production 
and consumer demand, which must be filled by more stable energy resources. Microgrids are localized grids managing energy for a specific region. While juggling output and demand, 
microgrid controllers require solutions to balance energy production and consumption. This paper addresses this need by proposing an AI-based load-forecasting approach for microgrids. 
Accurate load forecasting solutions consider factors impacting energy demand, including season, day type and location. Different ML techniques such as Fuzzy Logic Control (FLC), 
Artificial Neural Networks (ANNs), and Long Short-Term Memory (LSTM) networks can predict energy demand fluctuations and optimal energy production can be achieved without human intervention.

This solution will increase microgrid efficiency, reduce reliance on fossil fuels, and pave the way for a more sustainable energy future. 

After extensive research, I have composed a white paper which argues that ANN is the best power forecasting method. I have created a mini simulation of an ANN model, including a data preprocessing step which cleans the raw historical data and feature engineering that capture the cyclic temperature fluctuations in the climate data. 

**solar_predict.py** is a small-scale model which uses 5 weeks of weather data for the solar energy prediction and compares it to the recorded solar energy output. This model performed extremely well, achieving a total loss of 0.01 and a MAE of 0.06%.

**future_solar_predict.py** takes 3 years of detailed climate data, cleans the files, and reworks the datetime records. It then predicts the fourth year and compares it to the actual recorded solar output. This model is inefficient at its current standing, taking a long to run fully through. It has the potential to be further optimized. 

I created **generate_data.py** due to limitations in how much data I could fetch on visual crossing. Thus, generate_data.py writes 3 years of data including energy fluctuations. This data was extensive and sufficient for training my models. 


My hope is for this project to eventually work in real-time using a weather data API from VisualCrossing. 
