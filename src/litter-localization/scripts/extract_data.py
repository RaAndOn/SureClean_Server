#!/usr/bin/python

import rosbag
import os
import rospy
import cv2
import numpy as np

def save_image_to_file(image_msg,file_name):
        rospy.loginfo("Processing image...")
        # Image to numpy array
        img_arr = np.fromstring(image_msg.data, np.uint8).reshape(image_msg.height,image_msg.width)
        # img_arr = img_arr.reshape(image_msg.height,image_msg.width)
        print(img_arr.shape)

        # Decode to cv2 image and store
        #cv2_img = cv2.imdecode(img_arr, cv2.CV_LOAD_IMAGE_COLOR)
        cv2.imshow('Color image', img_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
				for topic, msg, t in bag_data.read_messages(topics=['/SurClean_UAV/annotated_image']):
					print(msg.image.encoding)
					save_image_to_file(msg.image,folder_path+'/'+bag+'.png')
				bag_data.close()

if __name__ == '__main__':
	try:
		extract_data()
	except rospy.ROSInterruptException:
		pass

