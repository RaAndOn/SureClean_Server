
import copy
import numpy as np
import cv2


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


def get_affine(points1, points2):
    # points1: (1, n1, 3) -- [x, y, id]
    # points2: (1, n2, 3) -- [x, y, id]
    return cv2.estimateRigidTransform(points1[:, :, :2], points2[:, :, :2], fullAffine=False)

if __name__ == "__main__":
    #### Create source and target points ####
    shape = (1, 5, 3)
    source = np.random.randint(0, 2249, shape).astype(np.int)
    theta = 30.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], 
                    [np.sin(theta), np.cos(theta)]]).astype(np.float)
    trans = np.array([1.0, -3.0]).astype(np.float)
    target = np.zeros(shape).astype(np.int)
    for i in range(shape[1]):
        point = source[0, i, :2]
        target[0, i, :2] = affine_transform(point, rot, trans, addNoise=True)
        target[0, i, 2] = i
        source[0, i, 2] = i
    #### Shuffle target points ####
    # jumbled = copy.deepcopy(target[0])
    # np.random.shuffle(jumbled)
    # target = np.array([jumbled])
    #### Compute affine transform and measure reprojection error ####
    # transformation = cv2.estimateRigidTransform(source[:, :, :2], target[:, :, :2], False)
    transformation = get_affine(source, target)
    rot = transformation[:, 0:2]
    trans = transformation[:, 2]
    reproj_target = np.zeros(shape)
    reproj_error = 0.0
    for i in range(shape[1]):
        point = source[0, i, :2]
        reproj_target[0, i, :2] = affine_transform(point, rot, trans, addNoise=False)
        reproj_target[0, i, 2] = i
        reproj_error += np.linalg.norm(reproj_target[0, i, :2] - target[0, i, :2])
    reproj_error /= shape[1]
    print(target)
    print(reproj_target)
    print("Reprojection error: ", reproj_error)

