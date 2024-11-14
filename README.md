## DEVELOPED BY: SHALINI K
## REGISTER NO: 212222240095
## DATE:

# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.

### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions

### PROGRAM:

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the Bitcoin dataset
data = pd.read_csv('coin_Bitcoin.csv')  # Update with the actual path to your Bitcoin dataset

# Convert the 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot the time series data
plt.plot(data.index, data['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Bitcoin Close Price Time Series')
plt.show()

# Function to check stationarity using ADF test
def check_stationarity(timeseries):
    # Drop missing values before applying ADF test
    timeseries = timeseries.dropna()
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

# Check stationarity of Bitcoin Close prices
check_stationarity(data['Close'].dropna())

# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data['Close'].dropna())
plt.show()
plot_pacf(data['Close'].dropna())
plt.show()

# Train-test split (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Define and fit the SARIMA model on training data
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Make predictions on the test set
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot the actual vs predicted values
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions for Bitcoin Close Price')
plt.legend()
plt.show()

```
### OUTPUT:

<table>
  <tr>
    <td style="width:50%">
      <img src="https://github.com/user-attachments/assets/79cbff1d-0f89-4fac-a72e-4bcc591ff05f" style="width:48%; height:auto;">
    </td>
    <td style="width:50%">
      <img src="https://github.com/user-attachments/assets/5105dcc2-ac2f-40ec-867f-1ea391209fa7" style="width:48%; height:auto;">
    </td>
  </tr>
  <tr>
    <td style="width:50%">
      <img src="https://github.com/user-attachments/assets/892e2c02-d89f-41c9-8c77-464d317276bb" style="width:48%; height:auto;">
    </td>
    <td style="width:50%">
      <img src="https://github.com/user-attachments/assets/7429051c-72e7-43f3-9e98-7c753236f0de" style="width:48%; height:auto;">
    </td>
  </tr>
  <tr>
    <td style="width:50%" colspan="2" align="center">
      <img src="https://github.com/user-attachments/assets/e25e62d2-2aa4-40e1-9631-c8e74346d734" style="width:48%; height:auto;">
    </td>
  </tr>
</table>




### RESULT:
Thus, the python program based on the SARIMA model is executed successfully.
