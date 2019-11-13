#!/usr/bin/env python

import cv2
import json
import os
import math
import numpy as np
import rosbag
import rospkg
import rospy
import sys
import utm
import yaml

# Set global variables
def get_params_dictionary():
	rospack = rospkg.RosPack()
	params_path = os.path.join(rospack.get_path('litter_localization'),"params/params.yaml")
	with open(params_path, 'r') as stream:
		try:
			params = (yaml.load(stream))
		except Exception as exc:
			print (exc)
	params["sureclean_server_path"] = rospack.get_path('litter_localization').replace('/src/litter_localization','')
	params["ros_bag_path"] = os.path.join(params["sureclean_server_path"],params["ros_bag_path"])
	params["detection_directory"] = os.path.join(params["sureclean_server_path"],"litter-detection")
	return params

def save_data_to_json(file_name, meta_data):
	data = {}
	for bag, flight_data in meta_data.iteritems():
		data[bag] = ({
			'Image': flight_data[0],
			'GPS':
				{
				'Latitude':  flight_data[1],
				'Longitude': flight_data[2],
				'Altitude':  flight_data[3],
				}
			,
			'Attitude':
				{
				'x': flight_data[4],
				'y': flight_data[5],
				'z': flight_data[6],
				'w': flight_data[7],
				}
			,
			'Height': flight_data[8],
			'Covariance': flight_data[9]
		})

	with open(file_name, 'w+') as outfile:
	    json.dump(data, outfile, indent=2)


def save_image_to_file(image_msg, image_path):
### This function takes in an array of numbers and saves it as an image to a file ###
	img_arr = np.fromstring(image_msg.data, np.uint8).reshape(
		image_msg.height, image_msg.width, 3)
	cv2.imwrite(image_path, img_arr)
	return image_path


def extract_data():
### This function process the uav bags, extracting the images and saving the meta data to a json file ###
	params = get_params_dictionary()
	bag_names = []
	meta_data = dict()

	for bag in os.listdir(params["ros_bag_path"]):
		bag_file_path = os.path.join(params["ros_bag_path"], bag)
		if '.bag' in bag:
			bag = bag.replace('.bag', '')
			meta_data[bag] = []
			bag_names.append(bag)


			bag_data = rosbag.Bag(bag_file_path)
			for topic, msg, t in bag_data.read_messages(topics = ['/SureClean_UAV/annotated_image']):
				#####################################################
				#####         Extract Data for Each Bag		  #######
				#####################################################
				image_path = save_image_to_file(msg.image, params["ros_bag_path"]+'/'+bag+'.png')
				meta_data[bag].append(image_path)
				meta_data[bag].append(msg.gps.latitude)
				meta_data[bag].append(msg.gps.longitude)
				meta_data[bag].append(msg.gps.altitude)
				meta_data[bag].append(msg.attitude.quaternion.x)
				meta_data[bag].append(msg.attitude.quaternion.y)
				meta_data[bag].append(msg.attitude.quaternion.z)
				meta_data[bag].append(msg.attitude.quaternion.w)
				meta_data[bag].append(msg.height.data)
				meta_data[bag].append(msg.gps.position_covariance)

			bag_data.close()
	
	json_file_path = os.path.join(params["sureclean_server_path"], params["extracted_data_path"])
	save_data_to_json(json_file_path, meta_data)
	return json_file_path



if __name__ == '__main__':
	try:
		extract_data()
	except rospy.ROSInterruptException:
		pass
