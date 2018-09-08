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
from matplotlib import pyplot
from matplotlib.pyplot import subplots, show
#####################################################################
# Prepare the data set
# Set a directory for where the data set is located
# Record starting time
start_time = time.time()
np.random.seed(1000)

df1 = import_data = pd.read_csv('G:/self_driving_car/dataset/cluster_label_data_MovieLens_final.csv', low_memory=False,
                               delimiter=',')
df1 = df1.drop(['release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
             'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance ', 'Sci-Fi', 'Thriller', 'War', 'Western', 'occupation','user id',
              'timestamp','x_coordinate','y_coordinate'], axis=1)

col_names = ['movie title','rating','x_coordinate',	'y_coordinate',	'Cluster_label', 'Cache_Probability']
df2 = import_data = pd.read_csv('G:/self_driving_car/dataset/data_to_cache.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
# We needs only probability value of each video to be cached in specific RSU.
# Appends probability value to original data frame
fig1 = plt.figure()
df1.age.plot.hist(bins=30, rwidth=0.8)
plt.xlabel = ("Age of users")
plt.ylabel = ("Movie watching count")
plt.grid(color='gray', linestyle='dashed')
fig1.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/users_histo.pdf", bbox_inches='tight')
plt.show()
fig2 = plt.figure()
labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']
df1['age_group'] = pd.cut(df1.age, range(0, 81, 10), right=False, labels=labels)
distage = df1.groupby('age_group').agg({'rating':[np.size,np.mean]})
colors=["y","lightgreen","dodgerblue","mediumturquoise",'violet', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.pie(distage['rating']['size'],startangle=90, shadow=True, labels=labels,colors=colors,explode=(0.5,0,0,0,0,0,0,0.1), autopct = '%1.1f%%',radius=2.2, textprops={'fontsize': 10})
plt.axis('equal')
plt.tight_layout()
fig2.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/users.pdf", bbox_inches='tight')
pyplot.show()
plt.show()

df1['Cache_Probability'] = pd.Series(df2['Cache_Probability'])

#####################################################################
#print(df1.head())
# Filtering by RSUs, where each RSU correspond to one cluster label and sort the videos based on Cache_Probability

# RSU 1
indices_1 = df1['Cluster_label'] == 0
RSU_1 = df1.loc[indices_1,:]
RSU_1 = RSU_1.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
print(RSU_1.head())
recommendations_RSU=RSU_1.head(7)
fig3, ax = subplots()
x = recommendations_RSU['movie title']
y = recommendations_RSU['Cache_Probability']
xn = range(len(x))
ax.bar(xn, y, color='lightblue', linewidth=3)
ax.plot(xn, y, color='blue', linewidth=3)
#plt.ylabel('Caching probability')
plt.xticks(xn, x,rotation=30)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(color='gray', linestyle='dashed')
fig3.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/cache_probabilities.pdf", bbox_inches='tight')
pyplot.show()
prediction_from_DC_RSU_1 = RSU_1.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_1.csv')
# RSU 2
indices_2 = df1['Cluster_label'] == 1
RSU_2 = df1.loc[indices_2, :]
RSU_2 = RSU_2.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
prediction_from_DC_RSU_2 = RSU_2.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_2.csv')
# RSU 3
indices_3 = df1['Cluster_label'] == 2
RSU_3 = df1.loc[indices_3, :]
RSU_3 = RSU_3.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
prediction_from_DC_RSU_3 = RSU_3.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_3.csv')

# RSU 4
indices_4 = df1['Cluster_label'] == 3
RSU_4 = df1.loc[indices_4, :]
RSU_4 = RSU_4.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
prediction_from_DC_RSU_4 = RSU_4.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_4.csv')

# RSU 5
indices_5 = df1['Cluster_label'] == 4
RSU_5 = df1.loc[indices_5, :]
RSU_5 = RSU_5.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
prediction_from_DC_RSU_5 = RSU_5.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_5.csv')

# RSU 6
indices_6 = df1['Cluster_label'] == 5
RSU_6 = df1.loc[indices_6, :]
RSU_6 = RSU_6.sort_values(by=['rating', 'Cache_Probability'], ascending=False)
recommendations_RSU=RSU_6.head(7)
prediction_from_DC_RSU_6 = RSU_6.to_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_6.csv')
