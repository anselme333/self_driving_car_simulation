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
import matplotlib.pyplot as plt;
from cvxpy import OSQP
from matplotlib.pyplot import subplots, show
plt.rcdefaults()
plt.style.use('ggplot')
import osqp
import cvxpy as cvx
import lfucache.lfu_cache as lfu_cache  # LFU replacement policy from https://github.com/laurentluce/lfu-cache
from RSU_3C import rsu1_caching_service, rsu2_caching_service, rsu3_caching_service,rsu4_caching_service, \
    rsu5_caching_service, rsu6_caching_service
from RSU_3C import communication_resources_rsu
import time
seaborn.set_style(style='white')
########################################################################################################################
# Load self driving car routing map
# Starting time
start_time = time.time()
np.random.seed(100)
# RSU settings
routing = import_data = pd.read_csv('G:/self_driving_car/dataset/rsu_routing_map.csv', low_memory=False, delimiter=',')
routing.drop(['Unnamed: 0'], axis=1, inplace=True)
List_route = routing["rsu"]
# Caching capacity of RSU in megabyte
number_rsu = 6
video_format0 = [".avi", ".mpg"]

# car settings

car_starting_point = List_route[1]
car_cache_capacity = 100000  # In terms of gagabyte
car_computation_capacity = 304510  # In terms of GHz (304,510 MIPS at 3.6 GHz)

car_information = pd.read_csv('G:/self_driving_car/dataset/self_driver_car_passengers.csv', low_memory=False, delimiter=',')

passenger = car_information["user_id"]
number_passenger = len(passenger)
# Recommended contents to cache in self-driving car
# Ref.  caching_self_driving_bus_recommendation.py
content_to_cache_car = pd.read_csv('G:/self_driving_car/dataset/caching_self_driving_bus_recommendation.csv',
                                   low_memory=False, delimiter=',')
content_to_cache_car.drop(['Unnamed: 0'], axis=1, inplace=True)
content_to_cache_car = content_to_cache_car.drop_duplicates(subset='movie title', keep='first')

movie_title_car = content_to_cache_car['movie title']
movies_size_car = []
video_format_car = []
for i in range(len(movie_title_car)):
    movies_sizecar0 = random.uniform(317.0, 750.0)
    movies_size_car.append(movies_sizecar0)
    if i % 2 == 0:
        video_format_car10 = video_format0[0]
        video_format_car.append(video_format_car10)
    else:
        video_format_car110 = video_format0[1]
        video_format_car.append(video_format_car110)
content_to_cache_car['movies_size'] = movies_size_car
content_to_cache_car['video_format'] = video_format_car
content_format_car =content_to_cache_car['video_format'].values
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


# Based on IEEE 802.11ac
# http://litepoint.com/whitepaper/80211ac_Whitepaper.pdf, qorvo-wifi-data-rates-channels-capacity-white-paper.pdf
def communication_resources_inside_car():
    distance_wifi_user = 3  # 2 meters
    WiFi_throughput_efficiency_factor = 0.9
    wireless_bandwidth = 160 # d the 802.11ac can utilize also 80 MHz and 160 MHz bandwidths
    maximum_theoretical_data_rate = 3466.8  # Mbit per second
    N = number_passenger  # Number of users
    spectrum_efficiency = (passenger_car_decision_variable * WiFi_throughput_efficiency_factor * wireless_bandwidth *
                           (maximum_theoretical_data_rate * 1/N)) / N
    return spectrum_efficiency

########################################################################################################################
# Caching Model


content_size_car = content_to_cache_car['movies_size'].values
# LFUCache
# https://github.com/laurentluce/lfu-cache

cache_car = lfu_cache.Cache()
movie_title_car = movie_title_car.values

# We assume that the car start its journal from RSU 1 to RSU 6 with one hours deriving between 2 nearby RSU.
car_speed_max = routing["max_speed"]
distance_between_rsu = routing["distance_between_rsu"]
car_starting_speed = car_speed_max[1]
driving_time = []
for i in range(len(distance_between_rsu)):
    time_left_rsu = distance_between_rsu[1]/car_speed_max[i]
    # convert hour to seconds
    time_left_rsu = time_left_rsu * 3600
    driving_time.append(time_left_rsu)


# Self_driving car downlaod the contents

# precaching
def cache_until_cache_full_car(content_cache, content_size, cache_capacity):
    cache_size = 0
    i = 0
    for video in content_size:
        cache_size += video
        if cache_size <= cache_capacity:
            cache_car.insert(content_cache[i], content_format_car[i])
            i += 1
    return cache_car, cache_size


cache_car, cached_video_size = cache_until_cache_full_car(movie_title_car, content_size_car, car_cache_capacity)

########################################################################################################################
# Car

transmission_car_rsu_delay = []


def car_computation_based_caching(video_size, video_format):
    cache_convert_car =1
    required_computation_avi_mpeg = 1024 # MIPS
    computation_allocation_car = car_computation_capacity * (cache_convert_car * required_computation_avi_mpeg)\
                                     / (video_size* required_computation_avi_mpeg)
    if computation_allocation_car <= car_computation_capacity:
        converting_time_car = (video_size * cache_convert_car * required_computation_avi_mpeg )/\
                              computation_allocation_car
        video_format = video_size * cache_convert_car * required_computation_avi_mpeg
    return cache_convert_car, computation_allocation_car, converting_time_car, video_format


def car_caching_service(video_title, video_size, video_format):
    # Initialization
    cache_convert_car = 0
    computation_allocation_car = 0
    converting_time_car = 0
    video_format = 0
    cache_hit_rsu = 0
    cache_hit_and_convert_rsu = 0
    computation_allocation_rsu = 0
    converting_time_rsu = 0
    cache_hit_dc = 0
    car_rsu_associa = 1
    content_hit = cache_car.access(video_title)
    transmission_rsu_dc_delay = 0
    percentage_radio_spectrum = 0
    rsu_car_connection = 0
    if content_hit == video_format:
        cache_hit_car = 1
        transmission_car_rsu_delay = 0
        rsu_radio_resource = 0
    elif content_hit == "":
        rsu_car_connection = 1
        # End self_driving_car needs communication resource  to communicate with RSU
        percentage_radio_spectrum = random.random()
        wireless_bandwidth, spectrum_efficiency = communication_resources_rsu()
        instantaneous_data = np.multiply(1, (percentage_radio_spectrum *
                                             spectrum_efficiency * wireless_bandwidth))
        print("instantaneous_data", instantaneous_data)
        rsu_radio_resource = wireless_bandwidth
        transmission_car_rsu_delay = (car_rsu_associa * 8388608 * video_size) / (10000000 * instantaneous_data)
        if i in range(len(driving_time)):
            if transmission_car_rsu_delay >=driving_time[0]:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc,\
                transmission_rsu_dc_delay = rsu1_caching_service(video_title, video_size, video_format)
            elif transmission_car_rsu_delay >=driving_time[1]:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc, \
                transmission_rsu_dc_delay = rsu2_caching_service(video_title, video_size, video_format)
            elif transmission_car_rsu_delay >= driving_time[2]:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc, \
                transmission_rsu_dc_delay = rsu3_caching_service(video_title, video_size, video_format)
            elif transmission_car_rsu_delay >= driving_time[3]:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc, \
                transmission_rsu_dc_delay = rsu4_caching_service(video_title, video_size, video_format)
            elif transmission_car_rsu_delay >= driving_time[4]:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc, \
                transmission_rsu_dc_delay = rsu5_caching_service(video_title, video_size, video_format)
            else:
                cache_hit_rsu, cache_hit_and_convert_rsu, computation_allocation_rsu, converting_time_rsu, cache_hit_dc, \
                transmission_rsu_dc_delay = rsu6_caching_service(video_title, video_size, video_format)
    else:
        cache_hit_car = 1
        cache_convert_car, computation_allocation_car, converting_time_car, video_format =\
            car_computation_based_caching(video_size, video_format)
        transmission_car_rsu_delay = 0
        rsu_radio_resource = 0
    return rsu_car_connection, percentage_radio_spectrum, cache_hit_car, cache_convert_car, computation_allocation_car, converting_time_car, rsu_radio_resource,\
           cache_hit_rsu, computation_allocation_rsu, cache_hit_and_convert_rsu,transmission_car_rsu_delay, \
           converting_time_rsu,cache_hit_dc, transmission_rsu_dc_delay, video_size
########################################################################################################################

# Demands from the passengers

cache_hit_car_array = []
cache_convert_car_array = []
computation_allocation_car_array =[]
converting_time_car_array = []
cache_hit_rsu_array = []
computation_allocation_rsu_array = []
cache_hit_and_convert_rsu_array = []
cache_hit_dc_array =[]
video_size_array = []
transmission_user_car_delay = []
rsu_radio_resource_vector = []
transmission_car_rsu_delay_vector =[]
Passenger_connection_variable = []
transmission_rsu_dc_delay_vector = []
converting_time_rsu_array = []
percentage_radio_spectrum_array = []
rsu_car_connection_array = []

for i in range(number_passenger):
    secure_random = random.SystemRandom()
    video_title = secure_random.choice(movie_title_car)
    video_format = secure_random.choice(content_format_car)
    content_title_row = content_to_cache_car.loc[content_to_cache_car['movie title'] == video_title]
    video_size = content_title_row['movies_size'].values

    # Connect to self_driving car
    Passenger_connection_variable.append(1)
    #check
    maximum_data_rate = communication_resources_inside_car()
    transmission_user_car_delay0 = ((Passenger_connection_variable[i] * video_size[0]) * 8388608) / (10000000 *
                                                                                                    maximum_data_rate)
    rsu_car_connection, percentage_radio_spectrum, cache_hit_car, cache_convert_car, computation_allocation_car, converting_time_car, rsu_radio_resource, \
    cache_hit_rsu, computation_allocation_rsu, cache_hit_and_convert_rsu, transmission_car_rsu_delay, \
    converting_time_rsu, cache_hit_dc, transmission_rsu_dc_delay, video_size = \
        car_caching_service(video_title, video_size[0], video_format)

    cache_hit_car_array.append(cache_hit_car)
    cache_convert_car_array.append(cache_convert_car)
    computation_allocation_car_array.append(computation_allocation_car)
    converting_time_car_array.append(converting_time_car)
    cache_hit_rsu_array.append(cache_hit_rsu)
    computation_allocation_rsu_array.append(computation_allocation_rsu)
    cache_hit_and_convert_rsu_array.append(cache_hit_and_convert_rsu)
    cache_hit_dc_array.append(cache_hit_dc)
    video_size_array.append(video_size)
    transmission_user_car_delay.append(transmission_user_car_delay0)
    rsu_radio_resource_vector.append(rsu_radio_resource)
    transmission_car_rsu_delay_vector.append(transmission_car_rsu_delay)
    transmission_rsu_dc_delay_vector.append(transmission_rsu_dc_delay)
    converting_time_rsu_array.append(converting_time_rsu)
    percentage_radio_spectrum_array.append(percentage_radio_spectrum)
    rsu_car_connection_array.append(rsu_car_connection)


cache_hit_car_array = np.array(cache_hit_car_array)
cache_convert_car_array = np.array(cache_convert_car_array)
computation_allocation_car_array = np.array(computation_allocation_car_array)
converting_time_car_array = np.array(converting_time_car_array)
cache_hit_rsu_array = np.array(cache_hit_rsu_array)
computation_allocation_rsu_array = np.array(computation_allocation_rsu_array)
cache_hit_and_convert_rsu_array = np.array(cache_hit_and_convert_rsu_array)
cache_hit_dc_array = np.array(cache_hit_dc_array)
video_size_array = np.array(video_size_array)
transmission_user_car_delay = np.array(transmission_user_car_delay)
rsu_radio_resource_vector = np.array(rsu_radio_resource_vector)
transmission_car_rsu_delay_vector = np.array(transmission_car_rsu_delay_vector)
transmission_rsu_dc_delay_vector = np.array(transmission_rsu_dc_delay_vector)
converting_time_rsu_array = np.array(converting_time_rsu_array)
Passenger_connection_variable = np.array(Passenger_connection_variable)
percentage_radio_spectrum_array = np.array(percentage_radio_spectrum_array)
rsu_car_connection_array = np.array(rsu_car_connection_array)


# Relationship between caching hit and zipf distribution
# https://en.wikipedia.org/wiki/Zipf%27s_law

number_demand = len(transmission_user_car_delay)
m = len(cache_hit_car_array)
array_one = np.ones(number_demand)
cache_miss = array_one - cache_hit_car_array
zipf_parameter = []
rank_parameter = 0.9
hit_ratio_zipf = []
zipf_cache = []

for i in range(6,  11):
    s = i/10  # i is from o.6 to 1.0
    zipf_parameter.append(s)
    for j in range(1, m):
        hit_ratio2 = cache_hit_car_array[j] / (cache_miss[j] + cache_hit_car_array[j]*j**rank_parameter)
        hit_ratio_zipf.append(hit_ratio2)
    hit_sum = sum(hit_ratio_zipf)
    num = i-6
    zipf_cache0 = 1/(rank_parameter**zipf_parameter[num] * hit_sum)
    zipf_cache.append(zipf_cache0)
    bins = sorted(zipf_cache)
x_pos = np.arange(len(zipf_parameter))
fig1 = plt.figure()
plt.bar(x_pos, bins, align='center', width=0.2, color='r', alpha=0.5)
plt.plot(x_pos, bins, 'r-', linewidth=3, linestyle='-', marker='^', markersize=10)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(x_pos, zipf_parameter)
plt.xlabel('Zipf parameter', fontsize=12)
plt.ylabel('Normalized cache hit', fontsize=12)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
fig1.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/"
             "self_driving_car_simulation/plots/cache_hit_car.pdf", bbox_inches='tight')

########################################################################################################################

number_demand = len(transmission_user_car_delay)
array_one = np.ones(number_demand)
array_zero = np.zeros(number_demand)
total_delay = transmission_user_car_delay + (cache_convert_car_array * converting_time_car_array) + \
             (array_one - (cache_convert_car_array + cache_hit_car_array)) * (transmission_car_rsu_delay_vector +
                                                                    (converting_time_rsu_array * cache_hit_and_convert_rsu_array)) + (array_one - (cache_hit_and_convert_rsu_array + cache_hit_rsu_array)*transmission_rsu_dc_delay_vector)

objective_function = total_delay


# Settings initial parameters
epsilon = 1e-12  # Epsilon Convergence condition
L_smooth = 0.01  # L-smooth constant
relaxation_threshold = 7.0
approx_err = []


def initialize_parameter():
    varrho0 = 1e-12
    t = 0  # iteration
    # Initial optimal variable at 0-th iteration  to zeros
    opt_q, opt_h, opt_varrho = np.zeros([number_demand, ]), np.zeros([number_demand, ]), np.zeros([number_demand, ])
    # Array of optimal variable and approximation error
    opt_val_q = []
    opt_val_h = []
    opt_val_varrho = []
    approx_err = []

    # Objective function as defined in equation 34
    opt_val_q.append(np.dot(total_delay, Passenger_connection_variable) + (varrho0 / 2) *
                     np.power(np.linalg.norm(Passenger_connection_variable - opt_q), 2))
    opt_val_h.append(np.dot(total_delay, Passenger_connection_variable) + (varrho0 / 2) *
                     np.power(np.linalg.norm(cache_hit_car_array - opt_h), 2))
    opt_val_varrho.append(np.dot(total_delay, Passenger_connection_variable) + (varrho0 / 2) *
                          np.power(np.linalg.norm(cache_convert_car_array - opt_varrho), 2))
    # Initial approximate error
    approx_err.append(np.Inf)
    return t, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, approx_err


def bs_mm(i, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, varrho):
    x = cvx.Variable(number_demand)  # Minimize over x:q
    y = cvx.Variable(number_demand)  # Minimize  y:h
    w = cvx.Variable(number_demand)  # Minimize z:varrho
    x_xo_norm = 0
    y_yo_norm = 0
    w_wo_norm = 0

    for t in range(number_demand):
        if t == i:
            x_xo_norm += (x - opt_q[t]) ** 2
            y_yo_norm += (y - opt_q[t]) ** 2
            w_wo_norm += (w - opt_h[t]) ** 2

        else:
            x_xo_norm += np.linalg.norm(Passenger_connection_variable[t] - opt_q[t]) ** 2
            y_yo_norm += np.linalg.norm(cache_hit_car_array[t] - opt_q[t]) ** 2
            w_wo_norm += np.linalg.norm(cache_convert_car_array[t] - opt_h[t]) ** 2

        # Form objective: proximal upper bound function, due to quadratic terms
        # Minimize  over x:q
        obj1 = cvx.Minimize(cvx.sum(objective_function[t] * x + (varrho / 2) * x_xo_norm))

        # Create constraints
        constraint1 = [percentage_radio_spectrum_array * x <= array_one, cached_video_size * x <= car_cache_capacity,
                       cvx.sum(cache_convert_car_array * cache_hit_car_array * computation_allocation_car_array) * x
                       <= car_computation_capacity, ((cache_convert_car_array * cache_hit_car_array) +
                                                        (array_one - (cache_convert_car_array * cache_hit_car_array))*(
                           cache_hit_and_convert_rsu_array + cache_hit_rsu_array)) * x <= array_one, x * (cache_convert_car_array + rsu_car_connection_array * (
                       array_one - cache_convert_car_array)) <= array_one, 0 <= x, x <= 1]

        # Form and solve problem.
        prob1 = cvx.Problem(obj1, constraint1)
        prob1.solve(solver=OSQP, max_iter=8000, verbose=True)
        opt_q = x.value
        opt_val_q.append(prob1.value)

        # Minimize  over y:h
        obj2 = cvx.Minimize(cvx.sum(objective_function[t] * y + (varrho / 2) * y_yo_norm))

        # Create constraints
        constraint2 = [percentage_radio_spectrum_array * y <= array_one, cached_video_size * y <= car_cache_capacity,
                       cvx.sum(cache_convert_car_array * cache_hit_car_array * computation_allocation_car_array) * y
                       <= car_computation_capacity, ((cache_convert_car_array * cache_hit_car_array) +
                                                     (array_one - (cache_convert_car_array * cache_hit_car_array)) * (
                                                         cache_hit_and_convert_rsu_array + cache_hit_rsu_array)) * y <= array_one,
                       y * (cache_convert_car_array + rsu_car_connection_array * (
                           array_one - cache_convert_car_array)) <= array_one, 0 <= y, y <= 1]

        # Form and solve problem
        prob2 = cvx.Problem(obj2, constraint2)
        prob2.solve(solver=OSQP, max_iter=8000, verbose=True)
        opt_h = y.value
        opt_val_h.append(prob2.value)

        # Minimize over z:varrho
        obj3 = cvx.Minimize(cvx.sum(objective_function[t] * w + (varrho / 2) * w_wo_norm))

        # Create constraints
        constraint3 = [percentage_radio_spectrum_array * w <= array_one, cached_video_size * w<= car_cache_capacity,
                       cvx.sum(cache_convert_car_array * cache_hit_car_array * computation_allocation_car_array) * w
                       <= car_computation_capacity, ((cache_convert_car_array * cache_hit_car_array) +
                                                     (array_one - (cache_convert_car_array * cache_hit_car_array)) * (
                                                         cache_hit_and_convert_rsu_array + cache_hit_rsu_array)) * w <= array_one,
                       w * (cache_convert_car_array + rsu_car_connection_array * (
                           array_one - cache_convert_car_array)) <= array_one, 0 <= w, w <= 1]
        print("total_delay")
        print(total_delay)

        # Form and solve problem.
        prob3 = cvx.Problem(obj3, constraint3)
        prob3.solve(solver=OSQP, max_iter=8000, verbose=True)
        opt_varrho = w.value
        opt_val_varrho.append(prob3.value)
        approx_err.append(np.abs((opt_val_varrho[-2] - opt_val_varrho[-1]) / opt_val_varrho[-2]))
        x_value_relaxation = opt_varrho
        return opt_q, opt_h, opt_varrho, x_value_relaxation, opt_val_q, opt_val_h,  opt_val_varrho, approx_err


def obj_bsum_rounding(i, opt_Rounding, rounded_decision_value, varrho):
    x = cvx.Variable()  # Minimize f over x
    x_xo_norm2 = 0

    # Maximum violation of communication, computational, and caching capacities
    communication00 = [a * b for a, b in zip(percentage_radio_spectrum_array, rounded_decision_value)]
    computation00 = [a * b for a, b in zip(computation_allocation_car_array, rounded_decision_value)]
    caching00 = cached_video_size * sum(rounded_decision_value)
    violation_communication0 = [a - b for a, b in zip(communication00, array_one)]
    violation_computation0 = sum(computation00) - car_computation_capacity
    violation_caching0 = caching00 - car_cache_capacity

    violation_communication = max(0, sum(violation_communication0))
    violation_computation = max(0, violation_computation0)
    violation_caching = max(0, violation_caching0)
    constant_violation = 1 / np.sqrt(number_demand)
    new_max_communication = [violation_communication + b for b in percentage_radio_spectrum_array]
    new_max_computation = violation_computation + car_computation_capacity
    new_max_caching = violation_caching + car_cache_capacity
    max_violation = constant_violation * (violation_caching + violation_computation + violation_communication)
    for t in range(number_demand):
            if t == i:
                x_xo_norm2 += (x - opt_q[t]) ** 2
            else:
                x_xo_norm2 += np.linalg.norm(Passenger_connection_variable[t] - opt_q[t]) ** 2

    obj1 = cvx.Minimize(cvx.sum(objective_function[t] * x + (varrho / 2) * x_xo_norm2 + max_violation))

    constraints = [percentage_radio_spectrum_array * x <= new_max_communication,
                   computation_allocation_car_array * x <= new_max_computation, x * cached_video_size <= new_max_caching,
                   ((cache_convert_car_array * cache_hit_car_array) + (array_one - (cache_convert_car_array *
                                                                                    cache_hit_car_array)) *
                    (cache_hit_and_convert_rsu_array + cache_hit_rsu_array)) * x <= array_one, x *
                   (cache_convert_car_array + rsu_car_connection_array * (
                       array_one - cache_convert_car_array)) <= array_one, 0 <= x, x <= 1]

    prob_rounding = cvx.Problem(obj1, constraints)
    prob_rounding.solve(verbose=True)
    # Retrieve optimal value
    opt_Rounding.append(prob_rounding.value)
    return opt_Rounding

########################################################################################################################


# BS-MM using Cyclic coordinate selection rule
# Get initial values
t, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, approx_err = initialize_parameter()

while np.any(approx_err[-1] > epsilon):
    for i in range(0, number_demand):
        varrho1 = 1e-12
        opt_q, opt_h, opt_varrho, x_value_relaxation_cyc,  opt_val_q, opt_val_h, opt_val_cy, approx_err = \
            bs_mm(i, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, varrho1)
        opt_val_cy[t] = opt_val_cy[t] - 1 / 2 * np.gradient(opt_val_cy)[t]
        opt_varrho = x_value_relaxation_cyc
        t += 1
        approx_err_cyc = approx_err
        opt_val_cyc = opt_val_cy
m_relax_cyc = len(x_value_relaxation_cyc)

for i in range(0, m_relax_cyc):
    if x_value_relaxation_cyc[i] >= relaxation_threshold:
        x_value_relaxation_cyc[i] = 1
    else:
        x_value_relaxation_cyc[i] = 0
    opt_Rounding_cyc = obj_bsum_rounding(i, opt_val_varrho, x_value_relaxation_cyc, varrho1)
    t += 1


########################################################################################################################

# BS-MM using Randomized coordinate selection rule
t, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, approx_err = initialize_parameter()
while np.any(approx_err[-1] > epsilon):
    for i in range(0, number_demand):
        varrho2 = 1e-12
        i = np.random.randint(0, number_demand)
        opt_q, opt_h, opt_varrho, x_value_relaxation_rand, opt_val_q, opt_val_h, opt_val_rand, approx_err =\
            bs_mm(i, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, varrho2)
        opt_val_rand[t] = opt_val_rand[t] - 1 / 2 * np.gradient(opt_val_rand)[t]

        t += 1
        opt_val_ran, approx_err_ran = opt_val_rand, approx_err
m_relax_rand = len(x_value_relaxation_rand)
for i in range(0, m_relax_rand):
    if x_value_relaxation_rand[i] >= relaxation_threshold:
        x_value_relaxation_rand[i] = 1
    else:
        x_value_relaxation_rand[i] = 0
    opt_Rounding_rand = obj_bsum_rounding(i, opt_val_varrho,  x_value_relaxation_rand, varrho2)
    opt_Rounding_rand[t] = opt_Rounding_rand[t] - 1 / 2 * np.gradient(opt_Rounding_rand)[t]
    t += 1

########################################################################################################################


# BS-MM using Gauss-Southwell coordinate selection rule

t, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, approx_err = initialize_parameter()

while np.any(approx_err[-1] > epsilon):
    for i in range(0, number_demand):
        varrho3 = 1e-12
        i = np.argmax(np.abs(objective_function + varrho3/2 * (Passenger_connection_variable - np.array(opt_q))))
        opt_q, opt_h, opt_varrho, x_value_relaxation_gso, opt_val_q, opt_val_h, opt_val_gou, approx_err = \
            bs_mm(i, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, varrho3)
        opt_val_gou[t] = opt_val_gou[t] - 1 / 2 * np.gradient(opt_val_gou)[t]
        t += 1
        opt_val_gso, approx_err_gso = opt_val_gou, approx_err
m_relax_gso = len(x_value_relaxation_gso)

for i in range(0, m_relax_gso):
    if x_value_relaxation_gso[i] >= relaxation_threshold:
        x_value_relaxation_gso[i] = 1
    else:
        x_value_relaxation_gso[i] = 0
    opt_Rounding_gso = obj_bsum_rounding(i, opt_val_varrho,  x_value_relaxation_gso, varrho3)
    t += 1

#############################################
fig1, ax1 = plt.subplots(figsize=(9, 6))
cyc_Rounding, = plt.plot(opt_Rounding_cyc, 'r-', linewidth=3, linestyle='--', marker='s')
gso_Rounding, = plt.plot(opt_Rounding_gso, 'g-', linewidth=3, linestyle='-', marker='x')
ran_Rounding, = plt.plot(opt_Rounding_rand, 'b-', linewidth=3, linestyle='--', marker='+')
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Total delay minimization (R)', fontsize=18)
plt.legend([cyc_Rounding, gso_Rounding, ran_Rounding],
               ['Cyclic', 'Gauss-Southwell', 'Randomized'], fancybox=True,
               fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(1, 30)
plt.ylim(500, 6000)
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 6))
cyc, = plt.plot(opt_val_cyc, 'r-', linewidth=3, linestyle='--', marker='s')
gso, = plt.plot(opt_val_gso, 'g-', linewidth=3, linestyle='-', marker='x')
ran, = plt.plot(opt_val_ran, 'b-', linewidth=3, linestyle='--', marker='+')
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Total delay minimization(NR)', fontsize=18)
plt.legend([cyc, gso, ran],
               ['Cyclic', 'Gauss-Southwell', 'Randomized'], fancybox=True,
               fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(1, 30)
plt.ylim(500, 6000)
plt.show()


fig3, ax = subplots()
plt.xlabel('Iterations', fontsize=12); plt.ylabel('Total delay minimization', fontsize=12)
alpha_list = [.1, 1, 10, 30]
alpha_idx = 0
gso = [None] * np.size(alpha_list)
for alpha in alpha_list:
    t, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, approx_err = initialize_parameter()
    while approx_err[-1] > epsilon:
        for i in range(0, number_demand):
            i = np.argmax(np.abs(objective_function + varrho1 / 2 * (Passenger_connection_variable - np.array(opt_q))))
            opt_q, opt_h, opt_varrho, x_value_relaxation_gso, opt_val_q, opt_val_h, opt_val_gou, approx_err = \
            bs_mm(i, opt_q, opt_h, opt_varrho, opt_val_q, opt_val_h, opt_val_varrho, alpha)
            opt_val_gou[t] = opt_val_gou[t] - 1 / 2 * np.gradient(opt_val_gou)[t]
            t += 1
    gso[alpha_idx], = plt.plot(opt_val_gou, '-', label=''r'$\alpha_j$ = ' + str(alpha), linewidth=3) #Display results
    alpha_idx += 1
plt.legend(handles=gso)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(500, 6000)
plt.xlim(1, 30)
fig3.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal4/simulation/"
             "self_driving_car_simulation/plots/total_delay_minimization.pdf", bbox_inches='tight')

plt.show()
integrality_gap_cyc = [a/b for a,b in zip(opt_Rounding_cyc,opt_val_cyc)]
print("integrality_gap_cyc",min(integrality_gap_cyc))
integrality_gap_gso = [a/b for a,b in zip(opt_Rounding_gso,opt_val_gso)]
print("integrality_gap_gso",min(integrality_gap_gso))
integrality_gap_rand = [a/b for a,b in zip( opt_Rounding_rand, opt_val_ran)]
print("integrality_gap_rand", min(integrality_gap_rand))