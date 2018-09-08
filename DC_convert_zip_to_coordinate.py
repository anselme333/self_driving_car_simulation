# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi-access Edge Computing
# Author: Anselme Ndikumana
# Python 3.6.4 : Anaconda custom (64-bit)
#####################################################################
# Loading the required libraries
import pandas as pd
import time
import numpy as np
import os
import seaborn as sns; sns.set()
from uszipcode import ZipcodeSearchEngine
#####################################################################
# Source of data_set
# Prepare the data_set: http://enhancedatascience.com/2017/04/22/building-recommender-scratch/
# Set a directory for where the data_set is located
# Starting time
start_time = time.time()
np.random.seed(10000)
os.chdir("G:/self_driving_car/dataset/ml-100k")
# column headers
data_cols = ['user id','movie id','rating','timestamp']
item_cols = ['movie id','movie title','release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
             'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama','Fantasy', 'Film-Noir',
             'Horror', 'Musical', 'Mystery', 'Romance ', 'Sci-Fi', 'Thriller', 'War', 'Western']
user_cols = ['user id','age','gender','occupation', 'zip code']

# Loading the data into pandas data frames
users = pd.read_csv('u.user', sep='|', names=user_cols, encoding='latin-1')
item = pd.read_csv('u.item', sep='|', names=item_cols, encoding='latin-1')
data = pd.read_csv('u.data', sep='\t', names=data_cols, encoding='latin-1')

# Create a merged data frame
df = pd.merge(pd.merge(item, data), users)
# Make zip code search engine
search = ZipcodeSearchEngine()

# Create a list for latitude and longitude coordinates
print(df.head())
x_coordinate1 = []
y_coordinate1 = []


def convert_zip_coordinate(data_frame):
    for x in data_frame:
        # search zip code
        zipcode = search.by_zipcode(x)
        # Find latitude and longitude coordinates and append them to the list
        x_coordinate1.append(zipcode.Latitude)
        y_coordinate1.append(zipcode.Longitude)
    return x_coordinate1, y_coordinate1
#####################################################################


data_frame_zip = df['zip code']
x_coordinate, y_coordinate = convert_zip_coordinate(data_frame_zip)
print(x_coordinate)
print(y_coordinate)
df["x_coordinate"] = x_coordinate
df["y_coordinate"] = y_coordinate
data_set_with_xy = df.to_csv('G:/self_driving_car/dataset/coordinates_data_MovieLens4.csv')
