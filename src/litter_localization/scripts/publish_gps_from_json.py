#!/usr/bin/env python

import extract_data as ext
import math
import json
import os
import rospkg
import utm
import rospy
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Empty
from sureclean_ugv_goal_ui.srv import GPSGoal
import yaml


class odom_struct:
    def __init__(self, x, y):
        odom = utm.from_latlon(x, y)
        self.x = odom[0]
        self.y = odom[1]


# Set global variables
params = ext.get_params_dictionary()
json_file_path = os.path.join(
    params["sureclean_server_path"], params["gps_json_path"])


def publish_gps_from_json(move_ugv=False):
    ### This function publishes the contents of a bag to a GPS topic###

    # Initialize node, wait for services
    rospy.init_node('ugv_gps_publisher')
    rospy.wait_for_service('/sureclean/ugv_1/server_gps_goal')
    rospy.wait_for_service('/sureclean/ugv_1/go_to_next_goal')

    server_gps_goal = rospy.ServiceProxy(
        '/sureclean/ugv_1/server_gps_goal', GPSGoal)
    go_to_next_goal = rospy.ServiceProxy(
        '/sureclean/ugv_1/go_to_next_goal', Empty)

    with open(json_file_path) as json_file:
        gps_data = json.load(json_file)
    gps_data_msg = rospy.wait_for_message(params["ugv_gps_topic"], NavSatFix)

    ordered_data = greedy_planning(gps_data, gps_data_msg)
    for data in ordered_data:
        print(data)
        # If bag covariance is larger than smallest covariance, skip bag
        try:
            coverage_side = 2*data["GPS"]["Variance"]
            odom = server_gps_goal(
                data["GPS"]["Latitude"], data["GPS"]["Longitude"], coverage_side)
            print("Latitude: ", data["GPS"]["Latitude"],
                  " Longitude: ", data["GPS"]["Longitude"], "Coverage Side: ", coverage_side)
            print("Odometry X: ", odom.odomX, " Odometry Y: ",
                  odom.odomY, "Coverage Side: ", odom.coverageSide)
        except rospy.ServiceException as e:
            print(e)
    try:
        odom = server_gps_goal(gps_data_msg.latitude,
                               gps_data_msg.longitude, 0.0)
        print("Base Latitude: ", gps_data_msg.latitude,
              " Base Longitude: ", gps_data_msg.longitude)
        print("Base Odometry X: ", odom.odomX,
              " Base Odometry Y: ", odom.odomY)
    except rospy.ServiceException as e:
        print(e)

    # GPS data
    gps_data = [[gps_data_msg.latitude, gps_data_msg.longitude]]

    if move_ugv:
        try:
            go_to_next_goal()
            print("UGV going to goals")
        except rospy.ServiceException as e:
            print(e)


def greedy_planning(gps_data, ugv_gps):
    data_set = []
    ordered_data = []
    num_trash = 0
    for _, data in gps_data.iteritems():
        data_set.append(data)
        num_trash += 1
    utm_cur = gps2_meter(ugv_gps.latitude, ugv_gps.longitude)
    for i in range(num_trash):
        gps_data_tuples = []
        for data in data_set:
            utm_gps = gps2_meter(
                data["GPS"]["Latitude"], data["GPS"]["Longitude"])
            dist = math.sqrt((utm_gps.x-utm_cur.x)**2 +
                             (utm_gps.y - utm_cur.y)**2)
            gps_data_tuples.append((data, dist))
        gps_data_tuples = sorted(gps_data_tuples, key=lambda gps: gps[1])
        nearest_trash = gps_data_tuples[0][0]
        ordered_data.append(nearest_trash)
        utm_cur = gps2_meter(
            nearest_trash["GPS"]["Latitude"], nearest_trash["GPS"]["Longitude"])
        data_set = data_set_remove(data_set, nearest_trash)

    return ordered_data


def data_set_remove(data_set, near_trash):
    new_set = []
    for data in data_set:
        if (data["GPS"]["Latitude"] != near_trash["GPS"]["Latitude"] or data["GPS"]["Longitude"] != near_trash["GPS"]["Longitude"]):
            new_set.append(data)
    return new_set


def gps2_meter(latitude, longitude):
    return odom_struct(latitude, longitude)


if __name__ == '__main__':
    try:
        publish_gps_from_json()
    except rospy.ROSInterruptException:
        pass
