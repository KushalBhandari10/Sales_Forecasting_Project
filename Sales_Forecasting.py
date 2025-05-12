import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose  
from prophet import Prophet

dates = pd.date_range(start='2023-01-01', periods=24, freq='ME')  
sales = np.random.randint(50, 200, size=24)  # Mock sales data  
df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Tasks
# Data Prep:
# Split data into train (first 18 months) and test (last 6 months).
D_train,D_test,S_train,S_test = train_test_split(df["Date"],df["Sales"],test_size=0.25,random_state=1,shuffle=False)
print("Train",D_train,"\nTest",D_test)
# Standardize the Sales column.
scaler = StandardScaler()
df["Sales"] = scaler.fit_transform(df[["Sales"]])
print(df)
# Time Series Analysis:
# Decompose the series into trend, seasonality, and residuals.
df = df.set_index("Date")
result = seasonal_decompose(df["Sales"],model="additive",period=12)
result.plot()
plt.show()
# Create lag features for 1 and 2 months.
df["Lag_1"] = df["Sales"].shift(1)
df["Lag_2"] = df["Sales"].shift(2)
df = df.dropna()
print("\nData with Lag Features:\n", df)
# Forecasting:
# Use Prophet to predict sales for the next 6 months.
df_prophet = df.reset_index().rename(columns={'Date': 'ds', 'Sales': 'y'})  
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=6,freq="ME")
forecast = model.predict(future)
model.plot(forecast)
plt.show()

print("\nForecasted Sales for Next 6 Months:\n", forecast[['ds', 'yhat']].tail(6))

