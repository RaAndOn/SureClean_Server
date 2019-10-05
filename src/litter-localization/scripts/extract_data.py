#!/usr/bin/python

import rosbag
import os
import rospy
import cv2
import numpy as np
import json
import pixel_to_meters as pxm


data = {}

def save_data_to_json(file_name,write_data):
	if not data:
		data['run'] = []
	data['run'].append({
	    'Image': write_data[0],
	    'GPS' : [
	    	{
	    	'Latitude':  write_data[1],
		    'Longitude': write_data[2],
		    'Altitude':  write_data[3],
	    	}
	    ],
	    'Attitude' : [ 
	    	{
	    	'x' : write_data[4],
	    	'y' : write_data[5],
	    	'z' : write_data[6],
	    	'w' : write_data[7],
	    	}
	    ],
	    'Height' : write_data[8],
	})

	with open(file_name, 'w+') as outfile:
	    json.dump(data, outfile, indent=2)

def save_image_to_file(image_msg,file_name):
        rospy.loginfo("Processing image...")
        img_arr = np.fromstring(image_msg.data, np.uint8).reshape(image_msg.height,image_msg.width,3)
        cv2.imwrite(file_name, img_arr)
        rospy.loginfo("Saved to: " + file_name)


def extract_data():
	rospy.init_node('extract_data', anonymous=True)
	file_path = os.getcwd() + '/ros_bags'
	for directory in os.listdir(file_path):
		folder_path = file_path + '/' + directory
		for bag in os.listdir(folder_path):
			bag_file_path = folder_path + '/' + bag
			if '.bag' in bag:
				bag = bag.replace('.bag','')
				bag_data = rosbag.Bag(bag_file_path)
				write_data = []
				for topic, msg, t in bag_data.read_messages(topics=['/SurClean_UAV/annotated_image']):	
					save_image_to_file(msg.image,folder_path+'/'+bag+'.png')
					write_data.append(directory+'/'+bag+'.png')
					write_data.append(msg.gps.latitude)
					write_data.append(msg.gps.longitude)
					write_data.append(msg.gps.altitude)
					write_data.append(msg.attitude.quaternion.x)
					write_data.append(msg.attitude.quaternion.y)
					write_data.append(msg.attitude.quaternion.z)
					write_data.append(msg.attitude.quaternion.w)
					write_data.append(msg.height.data)
					save_data_to_json(folder_path+'/'+directory+'.txt',write_data)
					##########################################################
					########     Obtain Litter Coordinates 		    ##########
					##########################################################
					# Get list of litter_coordinates from Atulya's Code

					pxm.execute_pixel_to_meters(msg.attitude.quaternion)
				bag_data.close()

if __name__ == '__main__':
	try:
		extract_data()
	except rospy.ROSInterruptException:
		pass
