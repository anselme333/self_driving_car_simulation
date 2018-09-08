# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi access Edge Computing
# Author: Anselme
# Python 3.6.4 :: Anaconda custom (64-bit)
########################################################################################################################
# Needed packages
from __future__ import division
import random
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
plt.style.use('ggplot')
import lfucache.lfu_cache as lfu_cache  # LFU replacement policy from https://github.com/laurentluce/lfu-cache
import time
seaborn.set_style(style='white')
########################################################################################################################
# Load self driving car routing map
# Starting time
start_time = time.time()
np.random.seed(10000)
# LFUCache
# https://github.com/laurentluce/lfu-cache

cache_dc = lfu_cache.Cache()
content_to_cache_dc = pd.read_csv('G:/self_driving_car/dataset/cluster_label_data_MovieLens4.csv',
                                  low_memory=False, delimiter=',')
content_to_cache_dc = content_to_cache_dc.drop_duplicates(subset='movie title', keep='first')

movie_title_dc = content_to_cache_dc['movie title']
movies_size_dc = []
video_format_dc = []
video_format0 = [".avi", ".mpg"]
for i in range(len(movie_title_dc)):
    movies_size_dc0 = random.uniform(317.0, 750.0)
    movies_size_dc.append(movies_size_dc0)
    if i % 2 == 0:
        video_format_car10 = video_format0[0]
        video_format_dc.append(video_format_car10)
    else:
        video_format_dc110 = video_format0[1]
        video_format_dc.append(video_format_dc110)
content_to_cache_dc['movies_size'] = movies_size_dc
content_to_cache_dc['video_format'] = video_format_dc
content_title_cached_dc = movie_title_dc.values
content_size_cached_dc = content_to_cache_dc['movies_size']
content_size_cached_dc = content_size_cached_dc.values
content_format_cached_dc = content_to_cache_dc['video_format']
content_format_cached_dc = content_format_cached_dc.values
i = 0
for video in content_size_cached_dc:
    cache_dc.insert(content_title_cached_dc[i], content_format_cached_dc[i])
    i += 1


def dc_caching_service( video_title, video_format):
    cache_dc.access(video_title)
    cache_hit_dc = 1
    return cache_hit_dc, video_format

