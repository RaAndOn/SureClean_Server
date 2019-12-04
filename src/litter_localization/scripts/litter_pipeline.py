#!/usr/bin/env python

import cv2
import evaluate_data as eva
import extract_data as ext
import geometry_msgs.msg
import json
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import pixel_to_utm as pxm
import rosbag
import rospkg
import rospy
import save_gps
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import utm


use_variable_coverage = True  # whether to use the variance for coverage size

use_clustering = True  # whether to use clustering or just pick best bag
use_biasing = True     # whether to use the mean of clusteirng and best bag


debug_clusters = True     # visualizes silhouette scores for different Ks
debug_silhouette = False   # visualizes points and cluster means
debug_comparison = False   # visualizes clustering vs. best_bag approach

hardcode_num_clusters = False
hardcoded_num_clusters = 5

OUTLIER_THRESH = 2e-5     # used to filter out obvious anomalous points
COV_THRESH = [0.1, 0.1]   # used to filter out high covariances

params = ext.get_params_dictionary()
sys.path.insert(1, params["detection_directory"])
import dynamic_color_thresh as dct
json_file_path = os.path.join(
    params["sureclean_server_path"], params["gps_json_path"])


def compute_avg_dists_from_cluster_center(X, cluster_centers, labels):
    dist_sum = 0.0
    for i in range(len(X)):
        p = np.array(X[i])
        cluster_center = np.array(cluster_centers[labels[i]])
        dist_sum += np.linalg.norm(p - cluster_center)
    return dist_sum / len(X)


def get_cluster_centers(X, num_clusters, slack):
    if (hardcode_num_clusters):
        kmeans = KMeans(n_clusters=hardcoded_num_clusters).fit(X)
        return kmeans.cluster_centers_.tolist()
    # Find the optimal K and get the cluster means
    best_sil_score = -1
    best_clusters = None
    best_labels = None
    sils = []
    min_k = max(2, num_clusters-slack)
    max_k = min(len(X)-1, num_clusters+slack)
    for n in range(min_k, max_k+1):
        kmeans = KMeans(n_clusters=n).fit(X)
        avg_dists = compute_avg_dists_from_cluster_center(
            X, kmeans.cluster_centers_, kmeans.labels_)
        sil_score = silhouette_score(X, kmeans.labels_, metric='euclidean')
        sils.append(sil_score)
        if (sil_score > best_sil_score):
            best_sil_score = sil_score
            best_clusters = kmeans.cluster_centers_
            best_labels = kmeans.labels_
    # below is purely visualization for debugging
    if (debug_silhouette):
        plt.scatter(range(min_k, max_k+1), sils)
        plt.title('Silhouette scores for different Ks')
        plt.show()
    return best_clusters.tolist(), best_labels.tolist()


def filter_points(X):
    dists = []
    new_X = []
    for i in range(len(X)):
        min_dist = 1e10
        for j in range(len(X)):
            if (i != j):
                p1 = np.array([X[i][0], X[i][1]])
                p2 = np.array([X[j][0], X[j][1]])
                dist = np.linalg.norm(p1-p2)
                dists.append(dist)
                if (dist < min_dist):
                    min_dist = dist
        if (min_dist < OUTLIER_THRESH):  # remove obvious outliers
            new_X.append([X[i][0], X[i][1]])
    return new_X


def clustering(filtered_X, num_clusters, slack):
    cluster_centers, labels = get_cluster_centers(
        filtered_X, num_clusters, slack)
    # below is purely visualization for debugging
    if (debug_clusters):
        x0 = []
        y0 = []
        for [latitude, longitude] in filtered_X:
            utm_coord = utm.from_latlon(latitude, longitude)
            x0.append(utm_coord[0])
            y0.append(utm_coord[1])
        mx = []
        my = []
        for [latitude, longitude] in cluster_centers:
            utm_coord = utm.from_latlon(latitude, longitude)
            mx.append(utm_coord[0])
            my.append(utm_coord[1])
        plt.scatter(x0, y0, c='k', marker='+', alpha=0.2)
        plt.scatter(mx, my, c='r', marker='X')
        axis_margin = 2
        plt.xlim(min(min(x0), min(mx))-axis_margin,
                 max(max(x0), max(mx))+axis_margin)
        plt.ylim(min(min(y0), min(my))-axis_margin,
                 max(max(y0), max(my))+axis_margin)
        plt.title('Raw points (black) and cluster means (red)')
        axes = plt.gca()
        axes.set_aspect('equal', 'box')
        plt.show()
    return cluster_centers, labels


def litter_pipeline(litter_count, slack):
    ### This function represents the main pipeline of the server  ###

    # Start Node
    rospy.init_node('litter_pipeline', anonymous=True)

    best_calculated_gps = []

    # Initialize variables
    litter_count = int(litter_count)
    minimum_covariance = [100, 100]
    litter_dict = dict()
    cluster_dict = dict()

    # Process ROS Bags
    flight_data_path = ext.extract_data()
    with open(flight_data_path) as json_file:
        flight_data = json.load(json_file)

    # Loop through all bag data
    for name, data in flight_data.iteritems():
        # Run image through the litter detection
        litter_dict[name] = dct.run_dynamic_color_thresh(
            data["Image"], params["sureclean_server_path"])[name]

        # Skip bag if the detection errors are greater than slack
        if abs(len(litter_dict[name])-litter_count) > slack:
            continue

        pixels_list = litter_dict[name]

        if (len(pixels_list) == 0):
            continue

        # Don't use data if covariance is a nan
        if math.isnan(data["Covariance"][0]):
            continue

        # Only use the data from the bag with the best covariance
        lat_long_covariance = np.array(
            [data["Covariance"][0], data["Covariance"][4]])

        is_best_bag = False
        if np.sum(lat_long_covariance) < np.sum(minimum_covariance):
            is_best_bag = True

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
        camera_utm = utm.from_latlon(
            data["GPS"]["Latitude"], data["GPS"]["Longitude"])

        # Calculate the GPS coordinates of the litter
        calculated_gps = []
        calculated_utms = np.zeros((len(pixels_list), 2))
        for i in range(len(pixels_list)):
            easting = camera_utm[0] + world_xy[i][1]
            northing = camera_utm[1] + world_xy[i][0]
            calculated_utms[i, 0] = easting
            calculated_utms[i, 1] = northing
            latitude, longitude = utm.to_latlon(
                easting, northing, camera_utm[2], camera_utm[3])
            calculated_gps.append([latitude[0], longitude[0]])

        if (np.sum(lat_long_covariance) < np.sum(COV_THRESH)):
            cluster_dict[name] = calculated_gps

        if (is_best_bag):
            best_calculated_gps = calculated_gps

    X = []
    for name in cluster_dict:
        for (latitude, longitude) in cluster_dict[name]:
            X.append([latitude, longitude])
    filtered_X = filter_points(X)  # removes obvious outliers

    cluster_centers, labels = clustering(filtered_X, litter_count, slack)

    # Bias cluster centers towards best bag
    new_clusters = []
    for c_center in cluster_centers:
        best_dist = 1e10
        best_gps = None
        for gps_loc in best_calculated_gps:
            dist = np.linalg.norm(np.array(c_center) - np.array(gps_loc))
            if (best_gps is None or dist < best_dist):
                best_dist = dist
                best_gps = gps_loc
        new_clusters.append(
            ((np.array(c_center) + np.array(best_gps))/2).tolist())

    # compute cluster variances
    num_clusters = len(cluster_centers)
    dists = [0.0 for _ in range(num_clusters)]
    num_points = [0 for _ in range(num_clusters)]
    for i in range(len(filtered_X)):
        label = labels[i]
        x = utm.from_latlon(filtered_X[i][0], filtered_X[i][1])
        cc = utm.from_latlon(
            cluster_centers[label][0], cluster_centers[label][1])
        dists[label] += np.linalg.norm(np.array(x[0],
                                                x[1]) - np.array(cc[0], cc[1]))
        num_points[label] += 1

    variances = dict()
    for i in range(len(dists)):
        variance = dists[i] / num_points[i]
        variances[i] = variance
        if use_variable_coverage:
            cluster_centers[i].append(variance)
            new_clusters[i].append(variance)
        if not use_variable_coverage:
            cluster_centers[i].append(-1.0)
            new_clusters[i].append(-1.0)

    if (debug_comparison):
        best_gps_lat = []
        best_gps_long = []
        for [latitude, longitude] in best_calculated_gps:
            utm_coord = utm.from_latlon(latitude, longitude)
            best_gps_lat.append(utm_coord[0])
            best_gps_long.append(utm_coord[1])
        cluster_lat = []
        cluster_long = []
        for [latitude, longitude] in cluster_centers:
            utm_coord = utm.from_latlon(latitude, longitude)
            cluster_lat.append(utm_coord[0])
            cluster_long.append(utm_coord[1])
        cluster_bias_lat = []
        cluster_bias_long = []
        for [latitude, longitude] in new_clusters:
            utm_coord = utm.from_latlon(latitude, longitude)
            cluster_bias_lat.append(utm_coord[0])
            cluster_bias_long.append(utm_coord[1])
        plt.scatter(best_gps_lat, best_gps_long, c='r')
        plt.scatter(cluster_lat, cluster_long, c='b')
        plt.scatter(cluster_bias_lat, cluster_bias_long, c='g')
        axis_margin = 2
        plt.xlim(min(min(best_gps_lat), min(cluster_lat))-axis_margin,
                 max(max(best_gps_lat), max(cluster_lat))+axis_margin)
        plt.ylim(min(min(best_gps_long), min(cluster_long))-axis_margin,
                 max(max(best_gps_long), max(cluster_long))+axis_margin)
        plt.title(
            'Best Bag (red) vs. Clustering (blue) vs. Clustering w/ bias (green)')
        axes = plt.gca()
        axes.set_aspect('equal', 'box')
        plt.show()

    print("Variance in meters: ", variances)
    print("Cluster centers in lat/long: ", cluster_centers)

    if (use_clustering and use_biasing):
        cluster_centers = new_clusters

    # print("\n")
    # print("Number of clusters: ", len(cluster_centers))
    # print("Clusters: ", cluster_centers)
    # print("\n")
    # print("Number of items in best bag: ", len(best_calculated_gps))
    # print("calculated_gps: ", best_calculated_gps)
    # print("Best GPS Bag: ", best_bag)
    # print("\n")

    if (use_clustering):
        save_gps.save_gps_to_json(json_file_path, cluster_centers)
    else:
        save_gps.save_gps_to_json(json_file_path, best_calculated_gps)


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            print("Usage: litter_pipeline.py litter_count slack")
        else:
            litter_pipeline(int(sys.argv[1]), int(sys.argv[2]))
    except rospy.ROSInterruptException:
        pass
