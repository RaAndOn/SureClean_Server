#!/usr/bin/env python

import extract_data as ext
import json
import os
import rosbag
import rospkg
import rospy
from sensor_msgs.msg import NavSatFix
import yaml

# Set global variables
params = ext.get_params_dictionary()
json_file_path = os.path.join(
    params["sureclean_server_path"], params["gps_json_path"])


def save_gps_from_topic():
    ### This function saves a single message from a gps topic to a bag ###
    # Start node
    rospy.init_node('uav_gps_listener')

    # Delete bag if one exists
    if os.path.exists(json_file_path):
        os.remove(json_file_path)

    # Wait for next message on topic
    gps_data_msg = rospy.wait_for_message(params["gps_topic"], NavSatFix)

    # Print details of GPS data
    rospy.loginfo("Latitude: %s", gps_data_msg.latitude)
    rospy.loginfo("Longitude %s", gps_data_msg.longitude)
    rospy.loginfo("Data Saved To: %s", json_file_path)

    # GPS data
    gps_data = [[gps_data_msg.latitude, gps_data_msg.longitude]]
    save_gps_to_json(json_file_path, gps_data)


def save_gps_to_json(file_path, gps_data):
    data = {}
    for i in range(len(gps_data)):
        data[i] = ({
            'GPS':
            {
                'Latitude':  gps_data[i][0],
                'Longitude': gps_data[i][1],
                'Variance': gps_data[i][2]
            }
        })

    with open(file_path, 'w+') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == '__main__':
    try:
        save_gps_from_topic()
    except rospy.ROSInterruptException:
        pass
