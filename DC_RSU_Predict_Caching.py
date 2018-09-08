# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi access Edge Computing
# Author: Anselme
# Python 3.6.4 :: Anaconda custom (64-bit)
# Comment: Running very well
#####################################################################
# Loading the required libraries
import os
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
#####################################################################
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
              'timestamp', 'age', 'gender', 'zip code', 'movie id'], axis=1)


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
df = df.astype('float32')
print(df.head())
#####################################################################

predictors = df.drop(['Cluster_label'], axis=1).as_matrix()
target = to_categorical(df.Cluster_label)
n_cols = predictors.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_cols,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(6, activation='softmax'))  # We have 6 RSUs
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(predictors, target, epochs=100, batch_size=32, validation_split=0.6)
model.save('content_to_cache_MovieLens4.h5')
y_prediction = model.predict(predictors)
# plot history
fig1 = plt.figure()
pyplot.plot(history.history['loss'], label='Training',linewidth=2, markersize=12)
pyplot.plot(history.history['val_loss'], label='Testing', linewidth=2, markersize=12)
plt.xlabel('Number of Epochs')
plt.ylabel('Cross entropy loss function')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(fancybox=True, title='')
plt.grid(color='gray', linestyle='dashed')
fig1.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/video_request_probabilities.pdf", bbox_inches='tight')
pyplot.show()
np.savetxt('G:/self_driving_car/dataset/data_to_cache.csv', y_prediction, delimiter=',')
duration = time.time() - start_time
print("Running Time:", duration)
