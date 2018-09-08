# Paper: Caching in Self-Driving Car: Deep Learning, Communication, and Computation Approaches in
# Multi-access Edge Computing
# Author: Anselme
# Python 3.6.4 : Anaconda custom (64-bit)
#####################################################################

import io
import googlemaps
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# source: https://stackoverflow.com/questions/17267807/python-google-maps-driving-time
# https://github.com/googlemaps/google-maps-services-python
# Key for googlemaps API can be optained on https://developers.google.com/api-client-library/python/start/get_started
gmaps = googlemaps.Client(key='') # The key is Google API Console key
now = datetime.now()

# Import RSU location

df = import_data = pd.read_csv('G:/self_driving_car/dataset/rsu_location.csv', low_memory=False, delimiter=',')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
print(df.head())
print(df.info())

cluster_x_coordinate = df.cluster_x_coordinate.values
cluster_y_coordinate = df.cluster_y_coordinate.values
print(cluster_x_coordinate)
print(cluster_y_coordinate)

# Routing_map

rsu1 = [cluster_x_coordinate[0], cluster_y_coordinate[0]]
rsu2 = [cluster_x_coordinate[1], cluster_y_coordinate[1]]
rsu3 = [cluster_x_coordinate[2], cluster_y_coordinate[2]]
rsu4 = [cluster_x_coordinate[3], cluster_y_coordinate[3]]
rsu5 = [cluster_x_coordinate[4], cluster_y_coordinate[4]]
rsu6 = [cluster_x_coordinate[5], cluster_y_coordinate[5]]

route = []
distance = []
conversion_factor_mile_km = 0.62137119
duration = []
speed = []
#####################################################################
# route 1
directions_result_RSU_1_2 = gmaps.directions(rsu1, rsu2, mode="driving",
                                     avoid="ferries", departure_time=now)

distance1 = directions_result_RSU_1_2[0]['legs'][0]['distance']['text']
distance1 =re.findall("\d+\,\d+", distance1)
distance1 = float(distance1[0].replace(',', ''))
# source: How to Use Python to Convert Miles to Kilometers
# https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/
distance1 = distance1/conversion_factor_mile_km
duration1 = directions_result_RSU_1_2[0]['legs'][0]['duration']['text']
duration1 = re.findall("\d+", duration1)
hours = int(duration1[0])
minutes = int(duration1[1])
duration1 = ((hours * 60) + minutes)/60


# source: Distance, speed and time
# http://www.bbc.co.uk/bitesize/standard/maths_i/numbers/dst/revision/1/
# speed = distance/time
speed1 = distance1/duration1
route.append("RSU_1_2")
distance.append(distance1)
duration.append(duration1)
speed.append(speed1)

#####################################################################
# route 2
directions_result_RSU_2_3 = gmaps.directions(rsu2, rsu3, mode="driving",
                                     avoid="ferries", departure_time=now)


distance2 = directions_result_RSU_2_3[0]['legs'][0]['distance']['text']
print("distance2:", distance2)
distance2 =re.findall("\d+\,\d+", distance2)
distance2 = float(distance2[0].replace(',', ''))
print("distance2:", distance2)
# source: How to Use Python to Convert Miles to Kilometers
# https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/
distance2 = distance2/conversion_factor_mile_km
duration2 = directions_result_RSU_2_3[0]['legs'][0]['duration']['text']
duration2 = re.findall("\d+", duration2)
hours2 = int(duration2[0])
minutes2 = int(duration2[1])
duration2 = ((hours2 * 60) + minutes2)/60


# source: Distance, speed and time
# http://www.bbc.co.uk/bitesize/standard/maths_i/numbers/dst/revision/1/
# speed = distance/time
speed2 = distance2/duration2
route.append("RSU_2_3")
distance.append(distance2)
duration.append(duration2)
speed.append(speed2)

#####################################################################
# route 3

directions_result_RSU_3_4 = gmaps.directions(rsu3, rsu4, mode="driving",
                                     avoid="ferries", departure_time=now)

distance3 = directions_result_RSU_3_4[0]['legs'][0]['distance']['text']
print("distance3:", distance3)
distance3 = re.findall("\d+\,\d+", distance3)
distance3 = float(distance3[0].replace(',', ''))
print("distance3:", distance3)
# source: How to Use Python to Convert Miles to Kilometers
# https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/
distance3 = distance3/conversion_factor_mile_km
duration3 = directions_result_RSU_3_4[0]['legs'][0]['duration']['text']
print("duration3:", duration3)
duration3 = re.findall("\d+", duration3)
duration3 = (42 * 60)/60


# source: Distance, speed and time
# http://www.bbc.co.uk/bitesize/standard/maths_i/numbers/dst/revision/1/
# speed = distance/time
speed3 = distance3/duration3
route.append("RSU_3_4")
distance.append(distance3)
duration.append(duration3)
speed.append(speed3)

#####################################################################
# route 4

directions_result_RSU_4_5 = gmaps.directions(rsu4, rsu5, mode="driving",
                                     avoid="ferries", departure_time=now)

distance4 = directions_result_RSU_4_5[0]['legs'][0]['distance']['text']
print("distance4:", distance4)
distance4 = re.findall("\d+\,\d+", distance4)
distance4 = float(distance4[0].replace(',', ''))
print("distance4:", distance4)
# source: How to Use Python to Convert Miles to Kilometers
# https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/
distance4 = distance4/conversion_factor_mile_km
duration4 = directions_result_RSU_4_5[0]['legs'][0]['duration']['text']
duration4 = (29 * 60)/60


# source: Distance, speed and time
# http://www.bbc.co.uk/bitesize/standard/maths_i/numbers/dst/revision/1/
# speed = distance/time
speed4 = distance4/duration4
route.append("RSU_4_5")
distance.append(distance4)
duration.append(duration4)
speed.append(speed4)

#####################################################################
# route 5

directions_result_RSU_5_6 = gmaps.directions(rsu5, rsu6, mode="driving",
                                     avoid="ferries", departure_time=now)

distance5 = directions_result_RSU_5_6[0]['legs'][0]['distance']['text']
print("distance5:", distance5)
distance5 = re.findall("\d+\,\d+", distance5)
distance5 = float(distance5[0].replace(',', ''))
print("distance5:", distance5)
# source: How to Use Python to Convert Miles to Kilometers
# https://www.pythoncentral.io/how-to-use-python-to-convert-miles-to-kilometers/
distance5 = distance5/conversion_factor_mile_km
duration5 = directions_result_RSU_5_6[0]['legs'][0]['duration']['text']
duration5 = (28 * 60)/60


# source: Distance, speed and time
# http://www.bbc.co.uk/bitesize/standard/maths_i/numbers/dst/revision/1/
# speed = distance/time
speed5 = distance5/duration5
route.append("RSU_5_6")
distance.append(distance5)
duration.append(duration5)
speed.append(speed5)

# driving hours Inter RSUs
transport_hour0 = 2
transport_hour = []
new_distance = []
for i in range(len(speed)):
    new_distance0 = transport_hour0 * speed[i]
    new_distance.append(new_distance0)
    transport_hour.append(transport_hour0)

routing_map = pd.DataFrame({"rsu": route, "distance_between_rsu": new_distance,"duration": transport_hour, "max_speed": speed})
data_set_with_clustering = routing_map.to_csv('G:/self_driving_car/dataset/rsu_routing_map.csv')

"""

# determine range to print based on min, max lat and lon of the data
margin = 2 # buffer to add to the range
lat_min = min(cluster_x_coordinate) - margin
lat_max = max(cluster_x_coordinate) + margin
lon_min = min(cluster_y_coordinate) - margin
lon_max = max(cluster_y_coordinate) + margin
m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color = 'white', lake_color='#46bcec')
# convert lat and lon to map projection coordinates
lons, lats = m(cluster_x_coordinate, cluster_y_coordinate)
# plot points as red dots
m.scatter(lons, lats, marker='o', color='red', zorder=5)
plt.show()
"""

