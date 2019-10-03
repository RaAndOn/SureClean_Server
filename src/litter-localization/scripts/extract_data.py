#!/usr/bin/python

import rosbag
import os
import rospy

def extract_data():
	rospy.init_node('extract_data', anonymous=True)
	rate = rospy.Rate(10) # 10hz
	while not rospy.is_shutdown():
		file_path = os.getcwd() + '/ros_bags'
		for directory in os.listdir(file_path):
			folder_path = file_path + '/' + directory
			for bag in os.listdir(folder_path):
				bag = folder_path + '/' + bag
				print(bag)
				if '.bag' in bag:
					bag_data = rosbag.Bag(bag)
					for topic, msg, t in bag_data.read_messages(topics=['SurClean_UAV/annotated_image']):
						print(msg)
					bag_data.close()

if __name__ == '__main__':
	try:
		extract_data()
	except rospy.ROSInterruptException:
		pass

