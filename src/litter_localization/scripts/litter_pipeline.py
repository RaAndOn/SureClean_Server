#!/usr/bin/env python

import cv2
import evaluate_data as eva
import extract_data as ext
import geometry_msgs.msg
import json
import math
import numpy as np
import os
import pixel_to_utm as pxm
import rosbag
import rospkg
import rospy
import save_gps
import sys
import utm

params = ext.get_params_dictionary()
sys.path.insert(1, params["detection_directory"])
import dynamic_color_thresh as dct
json_file_path = os.path.join(params["sureclean_server_path"], params["gps_json_path"])

def litter_pipeline(litter_count, slack):
### This function represents the main pipeline of the server  ###

    # Start Node
    rospy.init_node('litter_pipeline', anonymous=True)

    # Initialize variables
    litter_count = int(litter_count)
    minimum_covariance = [100,100]
    litter_dict = dict()

    # Process ROS Bags
    flight_data_path = ext.extract_data()
    with open(flight_data_path) as json_file:
        flight_data = json.load(json_file)
	
    # Loop through all bag data
    for name, data in flight_data.iteritems():
        # Run image through the litter detection
        litter_dict[name] = dct.run_dynamic_color_thresh(data["Image"],params["sureclean_server_path"])[name]

        # Skip bag if the detection errors are greater than slack
        if abs(len(litter_dict[name])-litter_count) > slack:
            continue
        pixels_list = litter_dict[name]

        # Only use the data from the bag with the best covariance
        lat_long_covariance = np.array([data["Covariance"][0],data["Covariance"][4]])
        if np.sum(lat_long_covariance) > np.sum(minimum_covariance):
            continue
        # Don't use data if covariance is a nan
        if math.isnan(data["Covariance"][0]):
            continue
        minimum_covariance = lat_long_covariance

        # Update the best bag name
        best_bag = name

        # Get meters to pixel relationship
        attitude = geometry_msgs.msg.Pose()
        attitude.orientation.x = data["Attitude"]["x"]
        attitude.orientation.y = data["Attitude"]["y"]
        attitude.orientation.z = data["Attitude"]["z"]
        attitude.orientation.w = data["Attitude"]["w"]
        world_xy = pxm.pixel_to_utm_translations(
            attitude, data["Height"], pixels_list)
        camera_utm = utm.from_latlon(data["GPS"]["Latitude"], data["GPS"]["Longitude"])
        

        # Calculate the GPS coordinates of the litter
        calculated_gps=[]
        calculated_utms = np.zeros((len(pixels_list), 2))
        for i in range(len(pixels_list)):
            easting = camera_utm[0] + world_xy[i][1]
            northing = camera_utm[1] + world_xy[i][0]
            calculated_utms[i,0] = easting
            calculated_utms[i,1] = northing
            latitude, longitude=utm.to_latlon(
                easting, northing, camera_utm[2], camera_utm[3])
            calculated_gps.append([latitude[0], longitude[0]])

    print("Best GPS Bag: ", best_bag )
    save_gps.save_gps_to_json(json_file_path,calculated_gps)


if __name__ == '__main__':
	try:
		if len(sys.argv) != 3:
			print("Usage: litter_pipeline.py litter_count slack")
		else:
			litter_pipeline(sys.argv[1],sys.argv[2])
	except rospy.ROSInterruptException:
		pass
