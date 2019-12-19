# Reference: https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/?#
# Reference: https://github.com/keras-team/keras/blob/master/examples/conv_lstm.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import math
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()
timesteps = 60
df = pd.read_csv("msft.us.txt").fillna(0)

# New data frame that selects our Date and Close columns 
df_select = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

# Add our columns date and close to our new data frame
run_through_data = True
while run_through_data:
    for i in range(0, len(df)):
        df_select['Date'][i] = df['Date'][i]
        df_select['Close'][i] = df['Close'][i]
        run_through_data = False
print('done converting data')

# Drops our date column because our data column isnt readable in our dataset format
df_select.index = df_select.Date
df_select.drop('Date', axis=1, inplace=True)
new_dataset = df_select.values

# Selects our 80% training and 20% for testing
training_data = new_dataset[:6386]
testing_data = new_dataset[6386:]

# Scales our data because scaling our data will help our mse and accuracy
scale = MinMaxScaler(feature_range=(0, 1))
scaling_data = scale.fit_transform(new_dataset)

# Append scaling data to our x and y values
x_train = []
y_train = []
for i in range(timesteps, len(training_data)):
    x_train.append(scaling_data[i-timesteps:i, 0])
    y_train.append(scaling_data[i, 0])      
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
 

# Building LSTM and train using our x train and y train
input_shape = (x_train.shape[1], 1)
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metric=['acc'])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Use past 60 values from the train data to predict values
values_used_to_predict = df_select[len(df_select) - len(testing_data) - timesteps:].values
values_used_to_predict = values_used_to_predict.reshape(-1, 1)
values_used_to_predict = scale.transform(values_used_to_predict)

# Append our scaled data to our x and y
x_test = []
y_test = []
for i in range(timesteps, values_used_to_predict.shape[0]):
    x_test.append(values_used_to_predict[i - timesteps:i, 0])
    y_test.append(values_used_to_predict[i, 0])
x_test = np.array(x_test)
y_test = np.array(y_test)

# predict with our model with our testing data 
reshape_x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predict_price = model.predict(reshape_x_test)
transform_price = scale.inverse_transform(predict_price)

# plot our training vs testing to see how well our lstm understood our previous values
train_data = df_select[:6386]
testing_data = df_select[6386:]
testing_data['Predictions'] = transform_price
plt.figure(figsize=(16, 8))
plt.plot(train_data['Close'],  color='black', label='Raw Training Dataset')
plt.plot(testing_data['Close'],  color='red', label='Raw Testing Dataset')
plt.plot(testing_data['Predictions'], color='green', label='Training Data' )
plt.title('LSTM Price Prediction')
plt.xlabel('Year')
plt.ylabel('Prices of the Stock')
plt.xticks(np.arange(0, 7982, 1300), df['Date'][0:7982:1300])
plt.legend()

# calculate error
mse = mean_squared_error(testing_data['Close'], transform_price)
mae = mean_absolute_error(testing_data['Close'], transform_price)
rmse = math.sqrt(mean_squared_error(testing_data['Close'], transform_price))
print('Mean Square Error: ', mse)
print('Mean Aboslute Error: ', mae)
print('Root Mean Squared Eror: ', rmse)
end = time.time()
elapsed = (end-start)
print('Elapsed time: ', elapsed)
