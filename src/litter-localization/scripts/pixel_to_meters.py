import rospy
import numpy as np
import math


def calculate_pixel_to_meters(pixel_x=0,pixel_y=0,world_z,intrinsics, extrinsics):
	[[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]] = np.matmul(intrinsics,extrinsics)

	world_x = ((b1 - pixel_x*b3)*(pixel_y*c3 - c2) - (b2-pixel_y*b3)*(pixel_x*c3-c1))*z/((b1 - pixel_x*b3)*(a2-pixel_y*a3)-(b2-pixel_y*b3)*(a1-pixel_x*a3))

	world_y = ((a1-pixel_x*a3)*(c3*pixel_y-c2) - (a2-pixel_y*a3)*(c3*pixel_x-c1))*z/((a1-pixel_x*a3)*(b2-b3*pixel_y) - (a2-pixel_y*a3)*(b1-b3*pixel_x))

	return (pixel_x/world_x,pixel_y/world_y)

def quat2mat(x,y,z,w)
	w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def build_extrinsics(quaternion):
	rtk_to_camera = np.eye(4,dtype=np.float)
	rtk_to_camera[0,3] = 0.03
	rtk_to_camera[1,3] = -0.05855
	rotation_to_camera = quat2mat(quaternion)
	rotation_to_camera = np.append(rotation_to_camera,[0,0,0],axix=1)
	rotation_to_camera = np.append(rotation_to_camera,[0,0,0,1],axis=0)
	altitude_to_camera = np.eye(4,dtype=np.float)
	altitude_to_camera[2,3] =  -0.08
	extrinsics = np.matmul(altitude_to_camera,np.matmul(rotation_to_camera,rotation_to_camera))
	return extrinsics

if __name__ == '__main__':
	try:
		# Get quaternion,height from json/rosbag
		# Get pixel_x/pixel_y from ros topic subscribe
		pixel_x = 0
		pixel_y = 0
		height = 20 # dummy data
		extrinsics = build_extrinsics(x=0.016926233803018848,y=-0.0007391055822455237,z=0.5876669701418246,w=0.8089255148260016)
		intrinsics = np.eye(4,dtype=np.float)[0:3,:]  #dummy data -- Get actual intrinsics from dji camera
		pixel_to_world = calculate_pixel_to_meters(pixel_x,pixel_y,height,intrinsics,extrinsics)
		# publish this on a topic
	except rospy.ROSInterruptException:
		pass
