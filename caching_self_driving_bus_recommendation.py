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
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
#####################################################################
# Prepare the data set
# Set a directory for where the data set is located
# Record starting time
start_time = time.time()
np.random.seed(1000)
col_names = ['binary_classification']
df_passenger = import_data = pd.read_csv('G:/self_driving_car/dataset/self_driver_car_passengers.csv', low_memory=False,
                               delimiter=',')


def passenger_in_car():
    number_passengers = len(df_passenger.genders)
    return number_passengers


passengers = passenger_in_car()
print("Number of passengers", passengers)

labels=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']
df_passenger['age_groups'] = pd.cut(df_passenger.age, range(0, 81, 10), right=False, labels=labels)
age_group = df_passenger.groupby(['age_groups', 'genders']).size().sort_values(ascending=False)[:8]

fig1, ax = subplots()
age_group.plot(kind="bar",title="",label="count")
plt.ylabel('Age group')
plt.grid(color='gray', linestyle='dashed')
fig1.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/"
             "self_driving_car_simulation/plots/self_driving_passenger.pdf", bbox_inches='tight')
plt.show()
#####################################################################
# Cluster 1

df_cluster1 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_1.csv', low_memory=False,
                               delimiter=',')

df_binary_class_1 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_1.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster1['binary_classification'] = pd.Series(df_binary_class_1['binary_classification'])
df_cluster1.drop(['Unnamed: 0'], axis=1, inplace=True)

# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster1['Recommendation'] = 0
df_cluster1['inDF1'] = 1
df_recommendation1 = pd.merge(df_cluster1, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation1['Recommendation'] = df_recommendation1['Recommendation'] + df_cluster1['inDF1'].fillna(0)
df_recommendation1.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)

recommendations_cluster_1 = df_recommendation1.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]
print(recommendations_cluster_1.head())
recommendations_cluster_1 = recommendations_cluster_1.head(8)
#####################################################################
# Cluster 2

df_cluster2 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_2.csv', low_memory=False,
                               delimiter=',')

df_binary_class_2 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_2.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster2['binary_classification'] = pd.Series(df_binary_class_2['binary_classification'])
df_cluster2.drop(['Unnamed: 0'], axis=1, inplace=True)

# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster2['Recommendation'] = 0
df_cluster2['inDF1'] = 1
df_recommendation2 = pd.merge(df_cluster2, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation2['Recommendation'] = df_recommendation2['Recommendation'] + df_cluster2['inDF1'].fillna(0)
df_recommendation2.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)

recommendations_cluster_2 = df_recommendation2.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]

#####################################################################
# Cluster 3

df_cluster3 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_3.csv', low_memory=False,
                               delimiter=',')

df_binary_class_3= pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_3.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster3['binary_classification'] = pd.Series(df_binary_class_3['binary_classification'])
df_cluster3.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster3['Recommendation'] = 0
df_cluster3['inDF1'] = 1
df_recommendation3 = pd.merge(df_cluster3, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation3['Recommendation'] = df_recommendation3['Recommendation'] + df_cluster3['inDF1'].fillna(0)
df_recommendation3.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_3 = df_recommendation3.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]


#####################################################################
# Cluster 4

df_cluster4 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_4.csv', low_memory=False,
                               delimiter=',')

df_binary_class_4 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_4.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster4['binary_classification'] = pd.Series(df_binary_class_4['binary_classification'])
df_cluster4.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster4['Recommendation'] = 0
df_cluster4['inDF1'] = 1
df_recommendation4 = pd.merge(df_cluster4, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation4['Recommendation'] = df_recommendation4['Recommendation'] + df_cluster4['inDF1'].fillna(0)
df_recommendation4.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_4 = df_recommendation4.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]

####################################################################
# Cluster 5

df_cluster5 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_5.csv', low_memory=False,
                               delimiter=',')

df_binary_class_5 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_5.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster5['binary_classification'] = pd.Series(df_binary_class_5['binary_classification'])
df_cluster5.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster5['Recommendation'] = 0
df_cluster5['inDF1'] = 1
df_recommendation5 = pd.merge(df_cluster5, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation5['Recommendation'] = df_recommendation5['Recommendation'] + df_cluster5['inDF1'].fillna(0)
df_recommendation5.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_5 = df_recommendation5.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]

####################################################################
# Cluster 6

df_cluster6 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_6.csv', low_memory=False,
                               delimiter=',')

df_binary_class_6 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_6.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster6['binary_classification'] = pd.Series(df_binary_class_5['binary_classification'])
df_cluster6.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster6['Recommendation'] = 0
df_cluster6['inDF1'] = 1
df_recommendation6 = pd.merge(df_cluster6, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation6['Recommendation'] = df_recommendation6['Recommendation'] + df_cluster6['inDF1'].fillna(0)
df_recommendation6.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_6 = df_recommendation6.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]

####################################################################
# Cluster 7

df_cluster7 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_7.csv', low_memory=False,
                               delimiter=',')

df_binary_class_7 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_7.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster7['binary_classification'] = pd.Series(df_binary_class_7['binary_classification'])
df_cluster7.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster7['Recommendation'] = 0
df_cluster7['inDF1'] = 1
df_recommendation7 = pd.merge(df_cluster7, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation7['Recommendation'] = df_recommendation7['Recommendation'] + df_cluster7['inDF1'].fillna(0)
df_recommendation7.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_7 = df_recommendation7.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]


####################################################################
# Cluster 8

df_cluster8 = pd.read_csv('G:/self_driving_car/dataset/car_kmean_cluster_8.csv', low_memory=False,
                               delimiter=',')

df_binary_class_8 = pd.read_csv('G:/self_driving_car/dataset/car_binary_cluster_8.csv', low_memory=False,
                               delimiter=',', header=None, names=col_names)
df_cluster8['binary_classification'] = pd.Series(df_binary_class_8['binary_classification'])
df_cluster8.drop(['Unnamed: 0'], axis=1, inplace=True)


# Find common rows of 2 data_frame
# source: https://stackoverflow.com/questions/30328187/find-common-rows-of-2-dataframe-for-2-columns
# and assign recommendation  row to 1, where the common values are found
df_cluster8['Recommendation'] = 0
df_cluster8['inDF1'] = 1
df_recommendation8 = pd.merge(df_cluster8, df_passenger, how='left', left_on=['age_group', 'gender'], right_on=['age_groups', 'genders'])
df_recommendation8['Recommendation'] = df_recommendation8['Recommendation'] + df_cluster8['inDF1'].fillna(0)
df_recommendation8.drop(['inDF1','user_id','genders','age_y',"age_groups"], axis=1, inplace=True)
recommendations_cluster_8 = df_recommendation8.sort_values(by=['rating', 'Cache_Probability','Recommendation'], ascending = False)[:8]

recommendations_joined_cluster = pd.concat([recommendations_cluster_1, recommendations_cluster_2,
                                            recommendations_cluster_3, recommendations_cluster_4,
                                            recommendations_cluster_5,recommendations_cluster_6,
                                            recommendations_cluster_7, recommendations_cluster_8])

recommendations_joined_cluster.to_csv('G:/self_driving_car/dataset/caching_self_driving_bus_recommendation.csv')
fig3, ax = subplots()
content_to_cache = recommendations_joined_cluster
most_rated=content_to_cache.groupby(['movie title','age_group', 'gender']).size().sort_values(ascending=False)[:8]
most_rated.plot(kind="bar",title="",label="count")
most_rated.plot(kind="bar",title="",label="count")
plt.grid(color='gray', linestyle='dashed')
plt.ylabel('Age group')
plt.xticks(rotation=60)
fig3.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/car_cache_recommendation.pdf", bbox_inches='tight')
pyplot.show()

