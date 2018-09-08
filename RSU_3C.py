# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi access Edge Computing
# Author: Anselme
# Python 3.6.4 :: Anaconda custom (64-bit)
########################################################################################################################
# Needed packages
from __future__ import division
import random
import math
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
plt.style.use('ggplot')
import lfucache.lfu_cache as lfu_cache  # LFU replacement policy from https://github.com/laurentluce/lfu-cache
from DC import dc_caching_service
import time
seaborn.set_style(style='white')
########################################################################################################################
# Load self driving car routing map
# Starting time
start_time = time.time()
np.random.seed(10000)
# RSU settings
# Caching capacity of RSU in gagabyte
number_rsu = 6
cache_capacity_rsu = random.sample(range(100000, 110000), number_rsu)
# computation capacity in GHz
# https://en.wikipedia.org/wiki/Instructions_per_second
computation_capacity_rsu = []
for j in range(number_rsu):
    # 304,510 MIPS at 3.6 GHz
    computation_capacity_rsu.append(random.uniform(304510, 304510))

# Recommended contents to cache at RSU
# Ref.  DC_RSU_Caching_Recommendation.py

content_to_cache_rsu1 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_1.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu1 = content_to_cache_rsu1.drop_duplicates(['movie title'],keep='first')

content_to_cache_rsu2 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_2.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu2 = content_to_cache_rsu2.drop_duplicates(['movie title'],keep='first')

content_to_cache_rsu3 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_3.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu3 = content_to_cache_rsu3.drop_duplicates(['movie title'],keep='first')

content_to_cache_rsu4 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_4.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu4 = content_to_cache_rsu4.drop_duplicates(['movie title'],keep='first')

content_to_cache_rsu5 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_5.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu5 = content_to_cache_rsu5.drop_duplicates(['movie title'],keep='first')

content_to_cache_rsu6 = pd.read_csv('G:/self_driving_car/dataset/prediction_from_DC_RSU_6.csv',
                                    low_memory=False, delimiter=',')
content_to_cache_rsu6 = content_to_cache_rsu6.drop_duplicates(['movie title'],keep='first')


# Average bit rate the size of a 2 hour movie will vary between 317 MB and 750 MB
movie_title1 = content_to_cache_rsu1['movie title']
video_format0 = [".avi", ".mpg"]
movies_size1 = []
video_format1 = []
for i in range(len(movie_title1)):
    movies_size0 = random.uniform(317.0, 750.0)
    movies_size1.append(movies_size0)
    if i % 2 == 0:
        video_format10=video_format0[0]
        video_format1.append(video_format10)
    else:
        video_format110 = video_format0[1]
        video_format1.append(video_format110)
content_to_cache_rsu1['movies_size'] = movies_size1
content_to_cache_rsu1['video_format'] = video_format1

movie_title2 = content_to_cache_rsu2['movie title']
movies_size2 = []
video_format2 = []
for i in range(len(movie_title2)):
    movies_size20 = random.uniform(317.0, 750.0)
    movies_size2.append(movies_size20)
    if i % 2 == 0:
        video_format20 = video_format0[0]
        video_format2.append(video_format20)
    else:
        video_format220 = video_format0[1]
        video_format2.append(video_format220)
content_to_cache_rsu2['movies_size'] = movies_size2
content_to_cache_rsu2['video_format'] = video_format2

movie_title3 = content_to_cache_rsu3['movie title']
movies_size3 = []
video_format3 = []
for i in range(len(movie_title3)):
    movies_size30 = random.uniform(317.0, 750.0)
    movies_size3.append(movies_size30)
    if i % 2 == 0:
        video_format30 = video_format0[0]
        video_format3.append(video_format30)
    else:
        video_format330 = video_format0[1]
        video_format3.append(video_format330)
content_to_cache_rsu3['movies_size'] = movies_size3
content_to_cache_rsu3['video_format'] = video_format3

movie_title4 = content_to_cache_rsu4['movie title']
movies_size4 = []
video_format4 = []
for i in range(len(movie_title4)):
    movies_size40 = random.uniform(317.0, 750.0)
    movies_size4.append(movies_size40)
    if i % 2 == 0:
        video_format40 = video_format0[0]
        video_format4.append(video_format40)
    else:
        video_format440 = video_format0[1]
        video_format4.append(video_format440)
content_to_cache_rsu4['movies_size'] = movies_size4
content_to_cache_rsu4['video_format'] = video_format4

movie_title5 = content_to_cache_rsu5['movie title']
movies_size5 = []
video_format5 = []
for i in range(len(movie_title5)):
    movies_size50 = random.uniform(317.0, 750.0)
    movies_size5.append(movies_size50)
    if i % 2 == 0:
        video_format50 = video_format0[0]
        video_format5.append(video_format50)
    else:
        video_format550 = video_format0[1]
        video_format5.append(video_format550)
content_to_cache_rsu5['movies_size'] = movies_size5
content_to_cache_rsu5['video_format'] = video_format5

movie_title6 = content_to_cache_rsu6['movie title']
movies_size6 = []
video_format6 = []
for i in range(len(movie_title6)):
    movies_size60 = random.uniform(317.0, 750.0)
    movies_size6.append(movies_size60)
    if i % 2 == 0:
        video_format60 = video_format0[0]
        video_format6.append(video_format60)
    else:
        video_format660 = video_format0[1]
        video_format6.append(video_format660)
content_to_cache_rsu6['movies_size'] = movies_size6
content_to_cache_rsu6['video_format'] = video_format6

########################################################################################################################
# Communication Model
car_rsu_decision_variable = 1
passenger_car_decision_variable = 1

# Gets car location and calculate communication resources
bandwidth_rsu_dc = np.random.randint(60, 70) # In terms of mbps
ground_distance = 40        # Distance between end-user and BS in terms of metres
path_loss_factor = 4        # Path loss exponents
transmission_power = 20.0   # Transmission power in term of dbm of each car
random_theta = np.random.uniform(0.0, 1.0, size=120)*2*np.pi
random_radius = ground_distance * np.sqrt(np.random.uniform(0.0, 1.0, size=120))
x = random_radius * np.cos(random_theta)
y = random_radius * np.sin(random_theta)
uniform_distance_points = [(x[i], y[i]) for i in range(120)]


def get_rsu_location():
    select_location = np.random.randint(10, 120)
    x_uniform = uniform_distance_points[select_location][0]
    y_uniform = uniform_distance_points[select_location][1]
    user_location = math.sqrt(x_uniform**2+y_uniform**2)
    return user_location


# Based on IEEE 802.11P
def communication_resources_rsu():
    distance_vehicle_rsu = get_rsu_location()
    # Fading
    sigma = 7   # Standard  deviation[dB]
    wireless_bandwidth = 10 # It provides 10 MHz channel bandwidth and up to 27-Mbps data transmission rate
    mu = 0  # Zero mean
    number_cars = 50
    tempx = np.random.normal(mu, sigma,  number_cars )
    x= np.mean(tempx)  # In term of dBm
    PL_0_dBm = 34  # In terms of dBm;
    PL_dBm = PL_0_dBm + 10 * path_loss_factor * math.log10(distance_vehicle_rsu / ground_distance) + x
    path_loss = 10 ** (PL_dBm / 10)  # [milli - Watts]
    channel_gain = transmission_power - path_loss
    channel_gain = float(channel_gain)
    spectrum_efficiency = car_rsu_decision_variable * wireless_bandwidth * math.log(1 + transmission_power *
                                                                                    channel_gain ** 2)
    return wireless_bandwidth, spectrum_efficiency



########################################################################################################################
# Caching Model

content_size_RSU1 = content_to_cache_rsu1['movies_size'].values
content_size_RSU2 = content_to_cache_rsu2['movies_size'].values
content_size_RSU3 = content_to_cache_rsu3['movies_size'].values
content_size_RSU4 = content_to_cache_rsu4['movies_size'].values
content_size_RSU5 = content_to_cache_rsu5['movies_size'].values
content_size_RSU6 = content_to_cache_rsu6['movies_size'].values

content_format_RSU1 = content_to_cache_rsu1['video_format'].values
content_format_RSU2 = content_to_cache_rsu2['video_format'].values
content_format_RSU3 = content_to_cache_rsu3['video_format'].values
content_format_RSU4 = content_to_cache_rsu4['video_format'].values
content_format_RSU5= content_to_cache_rsu5['video_format'].values
content_format_RSU6 = content_to_cache_rsu6['video_format'].values

# LFUCache
# https://github.com/laurentluce/lfu-cache

cache_rsu1 = lfu_cache.Cache()


def cache_until_cache_full_rsu1(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu1.insert(content_cache[i], content_format_RSU1[i])
        i += 1
    return cache_rsu1


movie_title1 = movie_title1.values
cached_content_rsu1 = cache_until_cache_full_rsu1(movie_title1, content_size_RSU1, cache_capacity_rsu[0])
# print(cached_content_rsu1)

cache_rsu2 = lfu_cache.Cache()


def cache_until_cache_full_rsu2(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu2.insert(content_cache[i], content_format_RSU2[i])
        i += 1
    return cache_rsu2


movie_title2 = movie_title2.values
cached_content_rsu2 = cache_until_cache_full_rsu2(movie_title2, content_size_RSU2, cache_capacity_rsu[1])
# print(cached_content_rsu2)

cache_rsu3 = lfu_cache.Cache()


def cache_until_cache_full_rsu3(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu3.insert(content_cache[i], content_format_RSU3[i])
        i += 1
    return cache_rsu3


movie_title3 = movie_title3.values
cached_content_rsu3 = cache_until_cache_full_rsu3(movie_title3, content_size_RSU3, cache_capacity_rsu[2])
# print(cached_content_rsu3)


cache_rsu4 = lfu_cache.Cache()


def cache_until_cache_full_rsu4(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu4.insert(content_cache[i], content_format_RSU4[i])
        i += 1
    return cache_rsu4


movie_title4 = movie_title4.values
cached_content_rsu4 = cache_until_cache_full_rsu4(movie_title4, content_size_RSU4, cache_capacity_rsu[3])
# print(cached_content_rsu4)


cache_rsu5 = lfu_cache.Cache()


def cache_until_cache_full_rsu5(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu5.insert(content_cache[i], content_format_RSU5[i])
        i += 1
    return cache_rsu5


movie_title5 = movie_title5.values
cached_content_rsu5 = cache_until_cache_full_rsu5(movie_title5, content_size_RSU5, cache_capacity_rsu[4])
# print(cached_content_rsu5)


cache_rsu6 = lfu_cache.Cache()


def cache_until_cache_full_rsu6(content_cache, content_size, cache_capacity):
    i = 0
    while sum(content_size[:i]) <= cache_capacity:
        cache_rsu6.insert(content_cache[i], content_format_RSU6[i])
        i += 1
    return cache_rsu6


movie_title6 = movie_title6.values
cached_content_rsu6 = cache_until_cache_full_rsu6(movie_title6, content_size_RSU6, cache_capacity_rsu[5])


########################################################################################################################
# RSU 1

def rsu1_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[0] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[0]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu1_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu1.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu1_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################
# RSU 2


def rsu2_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[1] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[1]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu2_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu2.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu2_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################
# RSU 3

def rsu3_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[2] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[2]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu3_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu3.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu3_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################

# RSU 4


def rsu4_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[3] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[3]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu4_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu4.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu4_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################

# RSU 5


def rsu5_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[4] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[4]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu5_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu5.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu5_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################
########################################################################################################################

# RSU 6


def rsu6_computation_based_caching(video_size, video_format):
    cache_hit_and_convert_rsu = 1
    required_computation_avi_mpeg = 1024  # MIPS
    computation_allocation_rsu = computation_capacity_rsu[5] * (cache_hit_and_convert_rsu * required_computation_avi_mpeg) \
                                      / (video_size * required_computation_avi_mpeg)

    if computation_allocation_rsu <= computation_capacity_rsu[5]:
        converting_time_rsu = (video_size * cache_hit_and_convert_rsu * required_computation_avi_mpeg)\
                          / computation_allocation_rsu
    return cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format


def rsu6_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    dc_rsu_associa = 1
    content_hit = cache_rsu6.access(video_title)
    if content_hit == video_format:
        cache_hit_rsu = 1
        transmission_rsu_dc_delay = 0
    elif content_hit == "":
        # We convert megabyte to bits
        transmission_rsu_dc_delay = (dc_rsu_associa * 8388608 * video_size) / (10000000 * bandwidth_rsu_dc)
        cache_hit_dc, video_format = dc_caching_service(video_title, video_format)
        cache_hit_rsu = 1
    else:
        transmission_rsu_dc_delay = 0
        cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, video_format =\
            rsu6_computation_based_caching(video_size, video_format)
        cache_hit_rsu = 0

    return cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
           transmission_rsu_dc_delay
########################################################################################################################