import cv2
import geometry_msgs.msg
import math
import numpy as np
import rospy
import tf
import utm


def undistort_pixels(u, v, intrinsics):
### Get undistorted values of pixels ###
    distortion_coefficients = np.array(
        [-0.122735, 0.100671, -0.000448, -0.004441, 0.000000], dtype=float)
    output = cv2.undistortPoints(np.array(
        [[[u, v]]], dtype=np.float), intrinsics, distortion_coefficients).tolist()
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx*output[0][0][0]+cx
    v = fy*output[0][0][1]+cy
    return u, v


def calculate_world_to_pixel_ratio(pixels_list, world_z, intrinsics, extrinsics):
### Solve equations of projection matrix to get UTM offsets of image pixels ###
    [[a1, b1, c1, d1], [a2, b2, c2, d2], [a3, b3, c3, d3]
     ] = np.matmul(intrinsics, extrinsics)

    world_xy = []
    for u, v in pixels_list:
        u,v = undistort_pixels(u,v,intrinsics)
        
        A = np.array([[a1-a3*u, b1-b3*u], [a2-a3*v, b2-b3*v]])
        b = np.array([[(c3*u-c1)*world_z+d3*u-d1],
                      [(c3*v-c2)*world_z+d3*v-d2]])

        world_xy.append(np.linalg.solve(A, b).tolist())
    
    return world_xy


def quaternionToYawMatrix(attitude):
### Take FLU to ENU quaternion of UAV and create Euler yaw matrix from it ###
    quaternion = (
    attitude.orientation.x,
    attitude.orientation.y,
    attitude.orientation.z,
    attitude.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2] + np.pi # Rotate by 180

    return np.array(
        [[np.cos(yaw), -np.sin(yaw), 0],
         [np.sin(yaw), np.cos(yaw), 0],
         [0, 0, 1]])


def build_extrinsics(attitude):
### Build extrinsics matrix for UAV ###
    extrinsics = quaternionToYawMatrix(attitude)

    # Appending the translation column to the extrinsics matrix
    extrinsics = np.append(extrinsics, [[0], [0], [0]], axis=1)
    extrinsics[0, 3] = 0.03
    extrinsics[1, 3] = -0.05855
    extrinsics[2, 3] = -0.08

    return extrinsics


def pixel_to_utm_translations(attitude, height, pixels_list):
### Calculate the pixel to UTM translations using image meta data ###
    try:

        ##########################################################
        #############		  Actual Data 		##################
        ##########################################################

        # Camera Matrix/Intrinsics obtained from https://github.com/usrl-uofsc/dji_gimbal_cam/blob/master/cfg/zenmuse_x3.yaml
        intrinsics = np.array([[778.284644, 0.000000, 642.166530], [
                              0.000000, 778.583237, 366.065267], [0.000000, 0.000000, 1.000000]], dtype=np.float)
        extrinsics = build_extrinsics(attitude)

        return calculate_world_to_pixel_ratio(pixels_list, height, intrinsics, extrinsics)

    except rospy.ROSInterruptException:
        pass