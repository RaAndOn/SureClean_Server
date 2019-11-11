#!/usr/bin/env python

import extract_data as ext
import json
import os
import rospkg
import rospy
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Empty
from sureclean_ugv_goal_ui.srv import GPSGoal
import yaml

# Set global variables
params = ext.get_params_dictionary()
json_file_path = os.path.join(params["sureclean_server_path"], params["gps_json_path"])

def publish_gps_from_json(move_ugv = False):
### This function publishes the contents of a bag to a GPS topic###

    # Initialize node, wait for services
    rospy.init_node('ugv_gps_publisher')
    rospy.wait_for_service('/sureclean/ugv_1/server_gps_goal')
    rospy.wait_for_service('/sureclean/ugv_1/go_to_next_goal')

    server_gps_goal = rospy.ServiceProxy('/sureclean/ugv_1/server_gps_goal', GPSGoal)
    go_to_next_goal = rospy.ServiceProxy('/sureclean/ugv_1/go_to_next_goal', Empty)

    with open(json_file_path) as json_file:
        gps_data = json.load(json_file)

    for name, data in gps_data.iteritems():
        print(data)
        # If bag covariance is larger than smallest covariance, skip bag
        try:
            odom = server_gps_goal(data["GPS"]["Latitude"],data["GPS"]["Longitude"],False)
            print( "Latitude: ",data["GPS"]["Latitude"], " Longitude: ",data["GPS"]["Longitude"])
            print("Odometry X: ", odom.odomX, " Odometry Y: ", odom.odomY)
        except rospy.ServiceException as e:
            print(e)

    # Wait for next message on topic
    gps_data_msg = rospy.wait_for_message(params["ugv_gps_topic"], NavSatFix)
    try:
        odom = server_gps_goal(gps_data_msg.latitude,gps_data_msg.longitude,True)
        print( "Base Latitude: ",gps_data_msg.latitude, " Base Longitude: ",gps_data_msg.longitude)
        print("Base Odometry X: ", odom.odomX, " Base Odometry Y: ", odom.odomY)
    except rospy.ServiceException as e:
        print(e)

    # GPS data
    gps_data = [[gps_data_msg.latitude,gps_data_msg.longitude]]

    if move_ugv:
        try:
            go_to_next_goal()
            print("UGV going to goals")
        except rospy.ServiceException as e:
            print(e)

if __name__ == '__main__':
    try:
        publish_gps_from_json()
    except rospy.ROSInterruptException:
        pass