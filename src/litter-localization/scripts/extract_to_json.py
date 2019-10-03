import rosbag
import os
import rospy

def extract_bag_data():
	rospy.init_node('extract_data', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
		file_path = 'ros_bags/Run'
		for i in len(os.listdir(file_path)):
			for bag in os.listdir(file_path+str(i)+'/'):
				if '.bag' in bag:
					bag_data = rosbag.Bag(bag)
					print(bag)

				for topic, msg, t in bag.read_messages(topics=['SurClean_UAV/annotated_image']):
					print(msg)
		bag.close()


if __name__ == '__main__':
	try:
		print('entered')
	    extract_bag_data()
    except rospy.ROSInterruptException:
        pass