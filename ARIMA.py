# Reference: https://www.kaggle.com/bogdanbaraban/ar-arima-lstm#ARIMA-model
# Reference: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import time
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import warnings
warnings.filterwarnings("ignore")

# time our code
start = time.time()

# read file
df = pd.read_csv("msft.us.txt").fillna(0)
df.head()

# 80,20 split
training = df[:6386]
testing = df[6386:]

# plot our raw data (close)
f = plt.figure(figsize=(16,8))
plt.title('Microsoft Closing Prices')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(df['Close'], color='Black', label='Training Data')
plt.plot(testing['Close'], color='red', label='Testing Data')
plt.xticks(np.arange(0,7982,1300), df['Date'][0:7982:1300]) 
plt.legend()
f.show()

# 80,20 split
train_data = df[:6386]
test_data = df[6386:]

train_data_c = train_data['Close'].values
test_data_c = test_data['Close'].values
 
# ARIMA model lets us predict future values based on our data's past values
loop_train_data = [x for x in train_data_c]
predictions = []
i = 0
for j in range(len(test_data_c)):
    model = ARIMA(loop_train_data, order=(5, 1, 0))
    model_fit = model.fit(disp= 0)
    fit_forecast = model_fit.forecast()
    fit = fit_forecast[0]
    predictions.append(fit)
    data = test_data_c[j]    
    loop_train_data.append(data)
# plot testing
n = plt.figure(figsize=(16,8))
plt.title('Testing Data')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(test_data_c, color='black')
n.show()


# plot prediction
m = plt.figure(figsize=(16,8))
plt.title('Predicting Data')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(predictions, color='red') 
m.show()

# plot side by side
o = plt.figure(figsize=(16,8))
plt.title('Predicting Data vs Testing')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(test_data_c, color='black', label='Testing')
plt.plot(predictions, color='red', label='Predict')
plt.legend()
o.show()

# plots our alogrithms predicted prices vs our real raw data 
plt.figure(figsize=(16,8))
plt.plot(df['Close'], color='black', linestyle='dashed')
plt.plot(test_data.index, test_data['Close'], color='black', label='Raw Dataset Price')
plt.plot(test_data.index, predictions, color='red', marker='o', linestyle='dashed', label='Predicted Dataset Price')
plt.title('ARIMA Price Prediction')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.xticks(np.arange(0,7982, 1300), df['Date'][0:7982:1300])
plt.legend()


# calculate error
mse = mean_squared_error(test_data_c, predictions)
mae = mean_absolute_error(test_data_c, predictions)
rmse = math.sqrt(mean_squared_error(test_data_c, predictions))
print('Mean Square Error: ', mse)
print('Mean Aboslute Error: ', mae)
print('Root Mean Squared Eror: ', mse)
end = time.time()
print('Time elapsed: ', end-start)
