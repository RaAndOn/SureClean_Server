#!/usr/bin/python

import cv2
import extract_data as ext
import geometry_msgs.msg
import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pixel_to_utm as pxm
import rosbag
import rospy
import sys
import utm

params = ext.get_params_dictionary()
sys.path.insert(1, params["detection_directory"])
import dynamic_color_thresh as dct
json_file_path = os.path.join(params["sureclean_server_path"], params["gps_json_path"])


def evaluate_data():
### This is function is used to evaluate different data sets and litter localization methods ###

	# Start Node
	rospy.init_node('analyze_data', anonymous=True)
	litter_dict = dict()

    # Initialize variables
	calculated_utms = np.array([False])
	count = 0

    # Process ROS Bags
	flight_data_path = ext.extract_data()
	with open(flight_data_path) as json_file:
		flight_data = json.load(json_file)

	fig = plt.figure()
	fig.suptitle('Calculated Litter Positions', fontsize=20)
	plt.xlabel('Easting (m)', fontsize=18)
	plt.ylabel('Northing (m)', fontsize=18)
	
	# Loop through all bag data
	for name, data in flight_data.iteritems():
		# If bag covariance is larger than smallest covariance, skip bag
		if data["Covariance"][0] > 0.07 or data["Covariance"][4] > 0.07:
			print("has large covariance")
			print(data["Covariance"])
			continue
		# If bag has NAN covariance, skip it
		if math.isnan(data["Covariance"][0]):
			print("has NAN covariance")
			print(data["Covariance"])
			continue

		litter_dict[name] = dct.run_dynamic_color_thresh(data["Image"],params["sureclean_server_path"])[name]
		pixels_list = litter_dict[name]

		print("Bag: ", name,"GPS: ",data["GPS"]["Latitude"],data["GPS"]["Longitude"])
		print("Covariance: ", data["Covariance"])

        # Get meters to pixel relationship
		attitude = geometry_msgs.msg.Pose()
		attitude.orientation.x = data["Attitude"]["x"]
		attitude.orientation.y = data["Attitude"]["y"]
		attitude.orientation.z = data["Attitude"]["z"]
		attitude.orientation.w = data["Attitude"]["w"]
		world_xy = pxm.pixel_to_utm_translations(attitude, data["Height"], pixels_list)
		camera_utm = utm.from_latlon(data["GPS"]["Latitude"], data["GPS"]["Longitude"])

        # Calculate the GPS coordinates of the litter
		calculated_utms = np.empty((len(world_xy),2))
		calculated_gps = []
		for i in range(len(world_xy)):
			easting = camera_utm[0] + world_xy[i][1]
			northing = camera_utm[1] + world_xy[i][0]
			calculated_utms[i,0] = easting
			calculated_utms[i,1] = northing
			latitude, longitude=utm.to_latlon(
				easting, northing, camera_utm[2], camera_utm[3])
			calculated_gps.append([latitude, longitude])
		# Plot UTM estimates
		plot_calculated_utms(calculated_utms,fig,count)
		count += 1
	plt.show()

def standard_deviation_utm(calculated_utms):
### Calculate the standard deviation in meters of multiple UTMs estimates ###
	# Initialize variables
	image_count = calculated_utms.shape[2]
	litter_count = calculated_utms.shape[0]
	avg_utm = np.sum(calculated_utms,axis=2)/image_count

	s_squared = np.zeros(avg_utm.shape)
	for i in range(calculated_utms.shape[2]):
		s_squared +=  np.power(calculated_utms[:,:,i]-avg_utm,2)
	s = np.sqrt(s_squared/(calculated_utms.shape[2]-1))
	for i in range(litter_count):
		print("Standard Deviation Easting ", s[i,0])
		print("Standard Deviation Northing ", s[i,1])

def plot_calculated_utms(calculated_utms,fig = False,color_index = 0):
### Plot UTM values ### 
	if not fig:
		fig = plt.figure()
		fig.suptitle('Calculated Litter Positions', fontsize=20)
		plt.xlabel('Easting (m)', fontsize=18)
		plt.ylabel('Northing (m)', fontsize=18)
	marker_array = ['o','v','s','x','d','*','|','1']
	color_array = ['r','g','b','c','m','y','k','w']
	if len(calculated_utms.shape) > 2:
		for i in range(calculated_utms.shape[2]):
			for j in range(calculated_utms.shape[0]):
				marker = color_array[i] + marker_array[j]
				plt.plot(calculated_utms[j,0,i], calculated_utms[j,1,i],marker)
	else:
		for j in range(calculated_utms.shape[0]):
			marker = color_array[color_index] + 'o'
			plt.plot(calculated_utms[j,0], calculated_utms[j,1],marker)
	plt.axis('square')

if __name__ == '__main__':
	try:
		evaluate_data()
	except rospy.ROSInterruptException:
		pass
