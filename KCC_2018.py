# Paper: Joint  Communication, Computation, Caching, and Control in Big-Data Multi-access Edge Computing
# Author: Anselme Ndikumana
# Conference: KCC 2018
# Interpreter: python 3.6.4
#############################################################

from __future__ import division
#from pyspark.sql.functions import when
import random
import math
import seaborn as sns

import simpy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import colors as mcolors
sns.set_style("ticks")
# Load Data
startTime = time.time()
import_mec = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/IDEA2/dataset_youtube/'
                         'MEC_server_data.csv', low_memory=False, delimiter=',')
import_self_driving = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/IDEA2/dataset_youtube/'
                                  'self_driver_car_data.csv', low_memory=False, delimiter=',')
import_mec_cleaned = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/IDEA2/dataset_youtube/'
                                 'MEC_server_data_cleaned.csv', low_memory=False, delimiter=',')

import_self_driving_cleaned = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/'
                                          'IDEA2/dataset_youtube/self_driver_car_data_cleaned.csv', low_memory=False,
                                          delimiter=',')

data = pd.merge(import_mec, import_self_driving, how='inner', on=['Gender','Age_group'], left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)
#print(data.head(3))
data_sorted = data.sort_values('Number_people', ascending=False)
print(data_sorted)

# Return  movies which has up to 4 dominant categories of people in vehicle

data_cache = data_sorted.head(4)
print(data_cache)
df2 = data_cache.fillna(0)

# Clean data and keep only numerical values

df2 = data_cache.drop(['Gender', 'Age_group', 'Number_people'], axis=1)
data_cache_final = df2.sort_values(by=1, ascending=False, axis=1)
print(data_cache_final)

# Return the movies which have high value number of the reviewers

t = data_cache_final.max(axis=1)
#print("test", t)
t2 = t.to_dict()
#print("test", t2)
data_cache_final2 = data_cache_final.to_dict()
#print(data_cache_final2)
# Make a ranking table
df_result = pd.DataFrame(t, columns=['rank'])
print(df_result)
# make make ranking list
my_lank_list = df_result["rank"].tolist()
#print(my_lank_list)
length_rank = len(my_lank_list)

# Set up a function which grabs the column name which contains the ranking value from data_cache_final
#data_cache_final['column'] = data_cache_final.apply(lambda x: data_cache_final.columns[x.idxmax()], axis=1)


def get_col_name(row):
    b = (data_cache_final.ix[row.name] == 7.68)
    return b.index[b.argmax()]
# For each row, test which elements equal the ranking, and extract column name of a True. And apply it (row-wise)

#df_result.apply(get_col_name, axis=1)


caching_mov_recommendation = data_cache_final.idxmax(axis=1)
caching_mov_recommendation = pd.DataFrame(caching_mov_recommendation, columns=['video'])
print(caching_mov_recommendation)
# Graphs
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

NUM_COLORS = 20
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
clrs = sns.color_palette('husl', n_colors=NUM_COLORS)  # a list of RGB tuples
ax = import_mec_cleaned.set_index('Gender_Age').plot.bar(rot=0, color=clrs, fontsize=12)

plt.xlabel('Gender and age of consumers ', fontsize=12)
plt.ylabel('Number of demands for videos', fontsize=12)
plt.legend(loc='upper left', fancybox=True,fontsize=12)
#plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
change_width(ax, .1)
ax.figure.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/age_gender_content.pdf")
plt.show()

ax = import_self_driving_cleaned.set_index('Gender_Age').plot.bar(rot=0, legend=None, color=[plt.cm.Paired(np.arange(len(import_self_driving_cleaned)))], fontsize=12)
plt.xlabel('Gender and age of people in self-driving bus', fontsize=12)
plt.ylabel('Number of people in self-driving bus')
plt.grid(color='gray', linestyle='dashed')
plt.xticks(rotation=60, fontsize=12)
plt.yticks(fontsize=12)
change_width(ax, .3)
plt.show()
ax.figure.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/self_driving_car_simulation/plots/self_driving_car_kcc.pdf", bbox_inches='tight')

result0 = data_cache[['Gender', 'Age_group']]
result1 = pd.concat([result0, caching_mov_recommendation, df_result], axis=1)
result2 = result1['Gender'] = result1[['Gender', 'Age_group']].apply(lambda x: ''.join(x), axis=1)
result2 = pd.DataFrame(result2, columns=['Gender_Age'])
result3 = pd.concat([result2, caching_mov_recommendation, df_result], axis=1)
result3=result3.replace({'MALE': 'MALE_', 'FEMALE': 'FEMALE_'}, regex=True)
print("")
print(result3)
final_video = result3[['Gender_Age','video','rank']]
final_video_cache=final_video.groupby(['Gender_Age','video']).sum().sort_values('rank')
final_video_cache.unstack().plot(kind='bar',stacked=True, width=0.1)
plt.xlabel('Gender and age of targeted consumers', fontsize=16)
plt.ylabel('Recommended videos to cache (ranking)', fontsize=16)
plt.legend(loc='upper left', fancybox=True, title='Video and its ranking:',fontsize=12)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
change_width(ax, .1)
plt.show()
endTime = time.time()
simulation_time = endTime - startTime
print("Simulation time:", simulation_time, "seconds")


