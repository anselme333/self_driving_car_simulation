# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi access Edge Computing
# Author: Anselme
# Python 3.6.4 :: Anaconda custom (64-bit)
#####################################################################
# Loading the required libraries
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import LSTM
#####################################################################
# For clear implemention of MovieLens-Dataset---Rating-Prediction:
# https://github.com/AshwathSalimath/Movie-ratings-prediction-using-MovieLens-Dataset
# Prepare the dataset
# Set a directory for where the data set is located
# Record starting time
start_time = time.time()
np.random.seed(1000)

df = import_data = pd.read_csv('G:/self_driving_car/dataset/cluster_label_data_MovieLens_final.csv', low_memory=False,
                               delimiter=',')
#####################################################################
# Start by cleaning dataset
# To keep only needed features
df = df.drop(['release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
             'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance ', 'Sci-Fi', 'Thriller', 'War', 'Western', 'occupation','user id',
              'timestamp', 'age', 'gender', 'zip code', 'Cluster_label', 'movie id', 'movie title', 'x_coordinate',
              'y_coordinate' ], axis=1)
#####################################################################


def handle_non_numerical(df):
    columns = df.columns.values
    for column in columns:
        text_to_digital_value = {}

        def convert_to_int(val):
            return text_to_digital_value[val]
        if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
            column_contents = df[column].values.tolist()
            elements = set(column_contents)
            x = 0
            for i in elements:
                if i not in text_to_digital_value:
                    text_to_digital_value[i] = x
                    x += 1
            df[column] = list(map(convert_to_int, df[column]))
    return df


df = handle_non_numerical(df)

# Convert timestamp to date and time
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
print(df.head())


def series_to_supervised(data, n_in=1, n_out=1, dropnan =True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = df.values
rows, columns = df.shape
print("columns", columns)
# Integer encode direction
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])
# Ensure all data is float
values = values.astype('float32')
# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# Frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
# reframed.drop(reframed.columns[[1]], axis=1, inplace=True)
print(reframed.head())

#####################################################################
# Define and fit Model
# specify the number of lag hours
n_hours = 24
n_features = 1
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# split into train and test sets
values = reframed.values
# Training for one months
n_train_hours = 30 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

print("train_X.shape", train_X.shape)
print("train_y.shape", train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print("train_X.shape, train_y.shape, test_X.shape, test_y.shape")
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print("train_X.shape1",train_X.shape[1])
print("train_X.shape2",train_X.shape[2])

# design network
model = Sequential()
model.add(LSTM(42, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(42,return_sequences=True))
model.add(LSTM(42))
model.add(Dense(1, activation='relu'))
model.compile(loss='mae', optimizer='adam', metrics=['mae', 'acc'])
model.summary()
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y), shuffle=False)
model.save('video_Rating.h5')
# plot history
fig1 = plt.figure()
pyplot.plot(history.history['loss'], label='Training',linewidth=2, markersize=12)
pyplot.plot(history.history['val_loss'], label='Testing', linewidth=2, markersize=12)
plt.xlabel('Number of Epochs')
plt.ylabel('Mean Absolute Error')
plt.grid(color='gray', linestyle='dashed')
pyplot.legend()
fig1.savefig("MAE_plotting.pdf", bbox_inches='tight')
pyplot.show()

# make a prediction
y_prediction = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invert scaling for forecast
y_prediction_scare0 = concatenate((y_prediction, test_X[:, -0:]), axis=1)
y_prediction_scare1 = scaler.inverse_transform(y_prediction_scare0)
y_prediction_scare2 = y_prediction_scare1[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
y_actual_scare0 = concatenate((test_y, test_X[:, -0:]), axis=1)
y_actual_scare1 = scaler.inverse_transform(y_actual_scare0)
y_actual_scare2 = y_actual_scare1[:, 0]


fig2 = plt.figure(figsize=(15, 5))
# Plot and compare the two signals.
plt.plot(y_actual_scare2, label='true')
plt.plot(y_prediction_scare2, label='pred')
# Plot labels etc.
plt.ylabel("Video rating")
plt.legend()
fig2.savefig("video_rating_prediction.pdf", bbox_inches='tight')
plt.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(y_actual_scare2, y_prediction_scare2))
print('RMSE: %.3f' % rmse)
duration = time.time() - start_time
print("Running Time:", duration)
print(y_actual_scare2)
print(y_prediction_scare2)












