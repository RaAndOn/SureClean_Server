
import copy
import random
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

SIZE = 3

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


def isPotentialMatch(points1, points2, n):
    indices1 = random.sample(range(n), SIZE)
    indices2 = random.sample(range(n), SIZE)
    source = np.zeros((1, SIZE, 3))
    target = np.zeros((1, SIZE, 3))
    for i in range(SIZE):
        source[0, i, :2] = points1[0, indices1[i], :2]
        source[0, i, 2] = i
    for j in range(SIZE):
        target[0, j, :2] = points2[0, indices2[j], :2] 
        target[0, j, 2] = j
    transformation = cv2.estimateRigidTransform(np.array([source[0, :, :2]]), np.array([target[0, :, :2]]), fullAffine=False)
    return (transformation is not None), transformation, source, target


def compute_reproj_error(transformation, source, target):
    if (transformation is None):
        return sys.maxsize, None
    rot = transformation[:, :2]
    trans = transformation[:, 2]
    reprojections = np.zeros(source.shape)
    reproj_error = 0.0
    for i in range(source.shape[1]):
        reprojection = affine_transform(source[0, i, :2], rot, trans, addNoise=False)
        reprojections[0, i, :2] = reprojection
        reprojections[0, i, 2] = i
        reproj_error += np.linalg.norm(reprojection - target[0, i, :2])
    return reproj_error / float(source.shape[1]), reprojections


def get_affine(points1, points2):
    # points1: (1, n1, 3) -- [x, y, id]
    # points2: (1, n2, 3) -- [x, y, id]
    n1 = points1.shape[1]
    n2 = points2.shape[1]
    n = min(n1, n2)
    if (n1 < SIZE or n2 < SIZE):
        raise Exception("Not enough points to compute affine transform. Must have at least 3.")
    minReprojError = sys.maxsize
    bestTransform = None
    count = 0
    while (minReprojError > 5.0):
        _, transformation, source, target = isPotentialMatch(points1, points2, n)
        reprojError, _ = compute_reproj_error(transformation, source, target)
        if (minReprojError is None or reprojError < minReprojError):
            minReprojError = reprojError
            bestTransform = transformation
        count += 1
    return bestTransform, minReprojError


def plot_mapping(transformation, source, target):
    _, reproj = compute_reproj_error(transformation, source, target)
    if (reproj is None):
        print("No reprojections to plot.")
    plot_data = {
        0: {'c': 'tab:blue', 'm': 'o'}, 
        1: {'c': 'tab:orange', 'm': 'v'},
        2: {'c': 'tab:green', 'm': '<'},
        3: {'c': 'tab:red', 'm': 's'},
        4: {'c': 'tab:purple', 'm': 'p'},
        5: {'c': 'tab:brown', 'm': 'P'},
        6: {'c': 'tab:pink', 'm': 'x'},
        7: {'c': 'tab:gray', 'm': 'd'},
        8: {'c': 'tab:olive', 'm': 'D'},
        9: {'c': 'tab:cyan', 'm': '2'}
    }
    for i in range(source.shape[1]):
        plt.scatter(source[0, i, 0], source[0, i, 1], marker='o', color=plot_data[i]['c'])
        plt.scatter(target[0, i, 0], target[0, i, 1], marker='x', color=plot_data[i]['c'])
        plt.scatter(reproj[0, i, 0], reproj[0, i, 1], marker='+', color=plot_data[i]['c'])
    ax = plt.gca()
    ax.legend(['source', 'target', 'reprojection'])
    ax.set_xlim([-3000, 3000])
    ax.set_ylim([-3000, 3000])
    plt.show()
    


if __name__ == "__main__":
    #### Create source and target points ####
    shape = (1, 10, 3)
    source = np.random.randint(0, 2249, shape).astype(np.int)
    theta = 60.0 * np.pi / 180.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]]).astype(np.float)
    trans = np.array([-650.0, 300.0]).astype(np.float)
    target = np.zeros(shape).astype(np.int)
    for i in range(shape[1]):
        point = source[0, i, :2]
        target[0, i, :2] = affine_transform(point, rot, trans, addNoise=True)
        target[0, i, 2] = i
        source[0, i, 2] = i
    #### Shuffle target points ####
    orig_target = copy.deepcopy(target)
    jumbled = copy.deepcopy(target[0])
    np.random.shuffle(jumbled)
    target = np.array([jumbled])
    #### Compute affine transform and measure reprojection error ####
    transformation, reprojError = get_affine(source, target)
    print("Transform: ", transformation)
    print("Reprojection error: ", reprojError)
    plot_mapping(transformation, source, orig_target)

