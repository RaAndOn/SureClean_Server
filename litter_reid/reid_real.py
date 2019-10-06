
import json
import sys
import random

import cv2

import numpy as np
from matplotlib import pyplot as plt

NUMBER_OF_OBJECTS = 6

plot_data = {
    # C is in (B, G, R) format
    0: {'c': 'tab:blue', 'm': 'o', 'C': (255, 0, 0)}, 
    1: {'c': 'tab:orange', 'm': 'v', 'C': (0, 179, 255)},
    2: {'c': 'tab:green', 'm': '<', 'C': (0, 255, 0)},
    3: {'c': 'tab:red', 'm': 's', 'C': (0, 0, 255)},
    4: {'c': 'tab:purple', 'm': 'p', 'C': (255, 0, 255)},
    5: {'c': 'tab:brown', 'm': 'P', 'C': (0, 75, 107)},
    6: {'c': 'tab:pink', 'm': 'x', 'C': (239, 0, 255)},
    7: {'c': 'tab:gray', 'm': 'd', 'C': (100, 100, 100)},
    8: {'c': 'tab:olive', 'm': 'D', 'C': (14, 102, 35)},
    9: {'c': 'tab:cyan', 'm': '2', 'C': (255, 255, 0)}
}

def affine_transform(point, rot, trans, addNoise=False):
    # point: (2,) numpy array
    # rot: (2,2) numpy array
    # trans: (2,) numpy array
    eps = 10
    warped = rot @ point.T + trans
    if (addNoise):
        warped[0] = warped[0] + np.random.randint(-eps, eps)
        warped[1] = warped[1] + np.random.randint(-eps, eps)
    return warped

def isPotentialMatch(points1, points2):
    SIZE = 3
    n1 = points1.shape[1]
    n2 = points2.shape[1]
    n = min(n1, n2)
    if (n1 < SIZE or n2 < SIZE):
        raise Exception("Not enough points to compute affine transform. Must have at least " + SIZE + ".")
    indices1 = random.sample(range(n), SIZE)
    indices2 = random.sample(range(n), SIZE)
    source = np.zeros((1, SIZE, 2))
    target = np.zeros((1, SIZE, 2))
    for i in range(SIZE):
        source[0, i, :] = points1[0, indices1[i], :]
    for j in range(SIZE):
        target[0, j, :] = points2[0, indices2[j], :] 
    transformation = cv2.estimateRigidTransform(np.array([source[0, :, :]]), np.array([target[0, :, :]]), fullAffine=False)
    return (transformation is not None), transformation, source, target

def compute_reproj(transformation, source, target):
    if (transformation is None):
        return sys.maxsize, None
    rot = transformation[:, :2]
    trans = transformation[:, 2]
    reprojections = np.zeros(source.shape)
    reproj_error = 0.0
    for i in range(source.shape[1]):
        reprojection = affine_transform(source[0, i, :], rot, trans, addNoise=False)
        reprojections[0, i, :] = reprojection
        reproj_error += np.linalg.norm(reprojection - target[0, i, :])
    return reproj_error / float(source.shape[1]), reprojections

def get_affine(points1, points2):
    # points1: (1, n1, 2) -- [x, y]
    # points2: (1, n2, 2) -- [x, y]
    minReprojError = sys.maxsize
    bestTransform = None
    while (minReprojError > 1.0):
        _, transformation, source, target = isPotentialMatch(points1, points2)
        reprojError, _ = compute_reproj(transformation, source, target)
        if (minReprojError is None or reprojError < minReprojError):
            minReprojError = reprojError
            bestTransform = transformation
    return bestTransform, minReprojError

def reproject_points(src, transformation):
    rot = transformation[:, :2]
    trans = transformation[:, 2]
    reprojections = np.zeros(src.shape)
    for i in range(src.shape[1]):
        reprojections[0, i, :] = affine_transform(src[0, i, :], rot, trans, addNoise=False)
    return reprojections

def find_correspondences(reproj, target):
    # Assumes reproj and target have the same number of points
    # Assumes that these points are true correspondences
    num_matches = min(reproj.shape[1], target.shape[1])
    matches = np.zeros((num_matches, 2)).astype(np.int)
    for i in range(reproj.shape[1]):
        if (i > num_matches):
            break
        best_dist = sys.maxsize
        best_idx = None
        for j in range(target.shape[1]):
            dist = np.linalg.norm(reproj[0, i, :] - target[0, j, :])
            if (dist < best_dist):
                best_dist = dist
                best_idx = j
        matches[i, 0] = i
        matches[i, 1] = best_idx
    return matches

def extract_image_results(image_name, data):
    return np.array([data[image_name]])

def get_reference_image(data):
    max_detections = -1
    max_key = None
    for key in data:
        if (len(data[key]) > max_detections):
            max_detections = len(data[key])
            max_key = key
    return (max_key, max_detections)

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data

def plot_mapping(pts1, pts2, img1_name, img2_name, mapping):
    img1 = cv2.imread("../litter-detection/images/input/" + img1_name)
    img2 = cv2.imread("../litter-detection/images/input/" + img2_name)
    count = 0
    for [p1_idx, p2_idx] in mapping:
        p1 = pts1[0, p1_idx, :]
        p2 = pts2[0, p2_idx, :]
        cv2.circle(img1, tuple(p1), 50, plot_data[count]['C'], 3, 8, 0)
        cv2.circle(img2, tuple(p2), 50, plot_data[count]['C'], 3, 8, 0)
        count += 1
    cv2.imwrite("output/" + img1_name[:-4] + "_matches.jpg", img1)
    cv2.imwrite("output/" + img2_name[:-4] + "_matches.jpg", img2)


if __name__ == "__main__":
    data = read_json("../litter-detection/results.json")
    # (max_key, max_detections) = get_reference_image(data)
    img1_name = "DJI_0074.JPG"
    img2_name = "DJI_0079.JPG"
    src = extract_image_results(img1_name, data)
    target = extract_image_results(img2_name, data)
    transformation, reprojError = get_affine(src, target)
    reprojections = reproject_points(src, transformation)
    matches = find_correspondences(reprojections, target)
    plot_mapping(src, target, img1_name, img2_name, matches)
    
    