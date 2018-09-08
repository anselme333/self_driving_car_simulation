# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi access Edge Computing
# Author: Anselme
# Python 3.6.4 :: Anaconda custom (64-bit)
# Comment: Running very well
#####################################################################
# Loading the required libraries
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#####################################################################
# Prepare the data set
start_time = time.time()
np.random.seed(1000)


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


df_kmean = pd.read_csv('G:/self_driving_car/dataset/data_from_rsu_kmean.csv', low_memory=False, delimiter=',')
# Drop no needed columns
df_kmean.drop(['Cluster_label', 'Unnamed: 0','Unnamed: 0.1','zip code'], axis=1, inplace=True)
print(df_kmean.info())
print(df_kmean.head())
duration = time.time() - start_time
print("Running Time:", duration)

#####################################################################
# Source: https://www.kaggle.com/parthsuresh/binary-classifier-using-keras-97-98-accuracy


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=1, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    #  For Compiling model, the logarithmic loss function and the Adam gradient optimizer are used
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


prediction_var = ['rating']

for i in range(8):
    if i == 1:
        # Cluster 1
        indices_1 = df_kmean['Cluster_age_label'] == 0
        cluster_1 = df_kmean.loc[indices_1, :]
        X = cluster_1[prediction_var].values
        Y = cluster_1.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_1.csv', encoded_Y, delimiter=',')
        car_kmean_cluster1 = cluster_1.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_1.csv')

    elif i == 2:
        # Cluster 2
        indices_2 = df_kmean['Cluster_age_label'] == 1
        cluster_2 = df_kmean.loc[indices_2, :]
        X = cluster_2[prediction_var].values
        Y = cluster_2.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_2.csv', encoded_Y, delimiter=',')
        car_kmean_cluster2 = cluster_2.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_2.csv')
    elif i == 3:
        # Cluster 3
        indices_3 = df_kmean['Cluster_age_label'] == 2
        cluster_3 = df_kmean.loc[indices_3, :]
        X = cluster_3[prediction_var].values
        Y = cluster_3.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_3.csv', encoded_Y, delimiter=',')
        car_kmean_cluster3 = cluster_3.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_3.csv')

    elif i == 4:
        # Cluster 4
        indices_4 = df_kmean['Cluster_age_label'] == 3
        cluster_4 = df_kmean.loc[indices_4, :]
        X = cluster_4[prediction_var].values
        Y = cluster_4.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_4.csv', encoded_Y, delimiter=',')
        car_kmean_cluster4 = cluster_4.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_4.csv')

    elif i == 5:
        # Cluster 5
        indices_5 = df_kmean['Cluster_age_label'] == 4
        cluster_5 = df_kmean.loc[indices_5, :]
        X = cluster_5[prediction_var].values
        Y = cluster_5.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_5.csv', encoded_Y, delimiter=',')
        car_kmean_cluster5 = cluster_5.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_5.csv')
    elif i == 6:
        # Cluster 6
        indices_6 = df_kmean['Cluster_age_label'] == 5
        cluster_6 = df_kmean.loc[indices_6, :]
        X = cluster_6[prediction_var].values
        Y = cluster_6.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_6.csv', encoded_Y, delimiter=',')
        car_kmean_cluster6 = cluster_6.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_6.csv')
    elif i == 7:
        # Cluster 6
        indices_7 = df_kmean['Cluster_age_label'] == 6
        cluster_7 = df_kmean.loc[indices_7, :]
        X = cluster_7[prediction_var].values
        Y = cluster_7.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_7.csv', encoded_Y, delimiter=',')
        car_kmean_cluster7 = cluster_7.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_7.csv')
    else:
        # Cluster 8
        indices_8 = df_kmean['Cluster_age_label'] == 7
        cluster_8 = df_kmean.loc[indices_8, :]
        X = cluster_8[prediction_var].values
        Y = cluster_8.gender.values
        # Changing gender to numerical values using LabelEncoder.
        encoder = LabelEncoder()
        # Fit label encoder and return encoded labels
        encoder.fit(Y)
        # Transform labels to normalized encoding.
        encoded_Y = encoder.transform(Y)
        # Evaluate model using standardized dataset.
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=32, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
        print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
        np.savetxt('G:/self_driving_car/dataset/car_binary_cluster_8.csv', encoded_Y, delimiter=',')
        car_kmean_cluster8 = cluster_8.to_csv('G:/self_driving_car/dataset/car_kmean_cluster_8.csv')








