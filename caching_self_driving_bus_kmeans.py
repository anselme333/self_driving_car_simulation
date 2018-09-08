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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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


def loading_cnn_output():
    df_car = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/'
                      'self_driving_car_simulation/data/self_driver_car_data.csv', low_memory=False,
                      delimiter=',')
    df_car = handle_non_numerical(df_car)
    df_car = df_car.astype('float32')
    return df_car


def download_rsu_prediction(rsu):
    if rsu == 1:
        prediction_from_DC_RSU_1 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_1.csv',
                                               low_memory=False,delimiter=',')
        return prediction_from_DC_RSU_1
    elif rsu == 2:
        prediction_from_DC_RSU_2 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_2.csv',
                                               low_memory=False, delimiter=',')
        return prediction_from_DC_RSU_2
    elif rsu == 3:
        prediction_from_DC_RSU_3 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_3.csv',
                                               low_memory=False, delimiter=',')
        return prediction_from_DC_RSU_3
    elif rsu == 4:
        prediction_from_DC_RSU_4 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_4.csv',
                                               low_memory=False, delimiter=',')
        return prediction_from_DC_RSU_4

    elif rsu == 5:
        prediction_from_DC_RSU_5 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_5.csv',
                                               low_memory=False, delimiter=',')
        return prediction_from_DC_RSU_5
    elif rsu == 6:
        prediction_from_DC_RSU_6 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_6.csv',
                                               low_memory=False, delimiter=',')
        return prediction_from_DC_RSU_6
    else:
        print("Self-driving car is not  connected to cache-enabled RSU")


rsu = 1
# Recommendation workflow
# Step 1: Download prediction prom RSU
data_from_rsu0 = download_rsu_prediction(rsu)
print(data_from_rsu0.info())
print(data_from_rsu0.head())
data_from_rsu = handle_non_numerical(data_from_rsu0)
data_from_rsu = data_from_rsu.astype('float32')
# Step 2: Use k-means for age-based clustering
kmeans = KMeans(n_clusters=8)
age = data_from_rsu["age"]
age_group = data_from_rsu["age_group"]
age = list(zip(age, age_group))
kmeans.fit(age)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
fig_handle = plt.figure()
colors = ["b.", "g.", "r.", "c.", "m.", "k.", "y.", "w."]
label_age=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']
Cluster_label = []

for i in range(len(age)):
    k = labels[i]
    k = int(k)
    print("Age:", age[i], "label:", labels[i])
    Cluster_label.append(labels[i])
    lines = plt.plot(age[i][0], age[i][1], colors[k], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker="x", s=250, linewidths=10, zorder=10,alpha=0.5,
            label='Age based clustering')
plt.grid(color='gray', linestyle='dashed')
plt.xlabel('Age')
plt.ylabel('Age group')
plt.legend()
fig_handle.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/data_from_rsu.pdf", bbox_inches='tight')
plt.show()
data_from_rsu1 = download_rsu_prediction(rsu)
data_from_rsu1["Cluster_age_label"] = Cluster_label
data_set_with_clustering = data_from_rsu1.to_csv('G:/self_driving_car/dataset/data_from_rsu_kmean.csv')
duration = time.time() - start_time
print("Running Time:", duration)












