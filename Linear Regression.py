# Reference: https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/?#
# Reference: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import time 
import warnings
warnings.filterwarnings("ignore")
start = time.time()

# read file
df = pd.read_csv("msft.us.txt").fillna(0)
df.head()

d = plt.figure(figsize=(16, 8))
plt.plot(df["Open"])
plt.plot(df["High"])
plt.plot(df["Low"])
plt.plot(df["Close"])
plt.title('Microsoft Stock Overview')
plt.ylabel('Price of Stock at Open, High, Low, Close')
plt.xlabel('Year')
plt.xticks(np.arange(0,7982,1300), df['Date'][0:7982:1300])
plt.legend(['Open', 'High', 'Low', 'Close'], loc='lower right')
d.show()

# plots the training and testing data, split 80, 20
training = df[:6386]
testing = df[6386:]

train_data_c = training['Close'].values
test_data_c = testing['Close'].values

print('training',train_data_c)
print('testing', test_data_c)
f = plt.figure(figsize=(16, 8))
plt.title('Microsoft Closing Prices')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(df['Close'], color='Black', label='Training Data')
plt.plot(testing['Close'], color='Orange', label='Testing Data')
plt.xticks(np.arange(0,7982,1300), df['Date'][0:7982:1300]) 
plt.legend()
f.show()

# dates need to be encoded
le = preprocessing.LabelEncoder()
df['Date'] = le.fit_transform(df['Date'])

training = df[:6386]
testing = df[6386:]

# here we drop close in the x, so that we can use the other data 
# values to predict our close
x_train = training.drop('Close', axis=1) 
y_train = training['Close'] 
x_test = testing.drop('Close', axis=1)
y_test = testing['Close']

# using Linear Regression to predict future values
model = LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
testing['Predictions'] = 0
testing['Predictions'] = prediction
training.index = df[:6386].index
testing.index = df[6386:].index

# dates need to be decoded to be displayed
df['Date'] = le.inverse_transform(df['Date'])

# plot predicted data 
g = plt.figure(figsize=(16,8))
plt.title('Predicted Microsoft Closing Prices')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.plot(training['Close'], color='black', label='Training Data')
plt.plot(testing['Predictions'], marker='.', color='red', label='Predicted Data')
plt.legend()
plt.xticks(np.arange(0,7982,1300), df['Date'][0:7982:1300]) 
g.show()


# calculate error
mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
rmse = math.sqrt(mean_squared_error(y_test, prediction))
print('Mean Square Error: ', mse)
print('Mean Aboslute Error: ', mae)
print('Root Mean Squared Eror: ', mse)
end = time.time()
elasped = (end - start)
print('Time elasped: ', elasped)
