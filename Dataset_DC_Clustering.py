# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi-access Edge Computing
# Author: Anselme
# Python 3.6.4 : Anaconda custom (64-bit)
#####################################################################
# Loading the required libraries
import pandas as pd
import time
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
#####################################################################
# Starting time
start_time = time.time()
np.random.seed(10000)
# Import data
df = import_data = pd.read_csv('G:/self_driving_car/dataset/coordinates_data_MovieLens4.csv', low_memory=False, delimiter=',')
# Remove all the NaN values based on user coordinates
df = df.dropna(subset=['x_coordinate', 'y_coordinate'])
x_coordinate = df["x_coordinate"]
y_coordinate = df["y_coordinate"]
x_coordinate = np.array(x_coordinate)
y_coordinate = np.array(y_coordinate)
coordinate = list(zip(x_coordinate, y_coordinate))
kmeans = KMeans(n_clusters=6)
kmeans.fit(coordinate)
print("coordinate length", len(coordinate))
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print("centroids", centroids)
number_rsu_cluster = dict(Counter(labels))
# Group the users based on their locations, where in each location we have one RSU.
with open('G:/self_driving_car/dataset/clusters_information.csv', 'w') as f:
    for key in number_rsu_cluster.keys():
        f.write("%s,%s\n"%(key, number_rsu_cluster[key]))
fig_handle = plt.figure()
colors = ["g.", "m.", "c.", "y.", "b.", "k."]

Cluster_label = []
for i in range(len(coordinate)):
    k = labels[i]
    k = int(k)
    print("coordinate:", coordinate[i], "label:", labels[i])
    Cluster_label.append(labels[i])
    plt.plot(coordinate[i][0], coordinate[i][1], colors[k], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker="x", s=150, linewidths=10, zorder=10, label='Roadside Units')

plt.grid(color='gray', linestyle='dashed')
plt.xlabel('X coordinates')
plt.ylabel('Y coordinates')
plt.legend()
fig_handle.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/rsu_deployment.pdf", bbox_inches='tight')
#plt.title('Roadside Units Deployment')
plt.show()
cluster_x_coordinate = centroids[:, 0]
cluster_y_coordinate = centroids[:, 1]
rsu_location = df = pd.DataFrame({
    "cluster_x_coordinate": cluster_x_coordinate,
    "cluster_y_coordinate": cluster_y_coordinate})
data_set_with_clustering = rsu_location.to_csv('G:/self_driving_car/dataset/rsu_location.csv')
df["Cluster_label"] = Cluster_label
data_set_with_clustering = df.to_csv('G:/self_driving_car/dataset/cluster_label_data_MovieLens4.csv')
duration = time.time() - start_time
print("Running Time:", duration)


