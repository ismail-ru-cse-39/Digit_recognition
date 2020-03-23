# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:40:21 2020

@author: Ismail
"""
"""
This is a python program using Long Short Term Memory network to predict the stock price of apple 
"""
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

"""
Get the stock information
"""

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')

print(df)

plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price in USD($)')
plt.show()

#Create a new Dataframe with only 'Close clomn
data = df.filter(['Close'])
#convert the dataframe to numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#print(training_data_len)
#print(dataset)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#print(scaled_data)

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#Split the data
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

#Conver x_train and y_train to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

#ReShape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

#Build the LSTm model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)
"""
"""#Create the testing data set
#Create the testing dataset
#Create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: , :]
#Create the dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#Conver the data to numpy data
x_test = np.array(x_test)

#Reshape the data
x_test =  np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root squared error
rmse = np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)


#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model') 
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#print valid data

print(valid)

#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-02', end='2019-12-17')
#Create new data frame
new_df = apple_quote.filter(['Close'])
#Get the last 60 days closing price values and conver the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_day_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test =[]
#Append the past 60 days 
X_test.append(last_60_day_scaled)
#Conver to numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("Our Predit data: ")
print()
print(pred_price)

#Get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print("Actual data: ")
print() 
print(apple_quote2['Close'])


        