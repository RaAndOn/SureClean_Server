
import copy
import random
import json
import numpy as np


MAX_ELEMS = 10
SHAPE = (1, MAX_ELEMS, 3)
NUM_SETS = 6
THETAS = random.sample(range(-180, 180), NUM_SETS)
TX = random.sample(range(-500, 500), NUM_SETS)
TY = random.sample(range(-500, 500), NUM_SETS)


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


def create_data():
    # Create point sets
    obj = {}
    source = np.random.randint(0, 2250, SHAPE).astype(np.int)
    for i in range(SHAPE[1]):
        source[0, i, 2] = i
    obj[0] = source
    for j in range(NUM_SETS):
        theta = THETAS[j] * np.pi / 180.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]]).astype(np.float)
        trans = np.array([TX[j], TY[j]]).astype(np.float)
        target = np.zeros(SHAPE).astype(np.int)
        for i in range(SHAPE[1]):
            point = source[0, i, :2]
            target[0, i, :2] = affine_transform(point, rot, trans, addNoise=True)
            target[0, i, 2] = i
        np.random.shuffle(target[0])
        obj[j] = np.array([target]).tolist()
    with open('points.json', 'w') as json_file:
        json.dump(obj, json_file)


def read_data():
    with open('points.json') as f:
        data = json.load(f)
    for key in data:
        data[key] = np.asarray(data[key])
        print(data[key])
    return data

if __name__ == "__main__":
    # create_data()
    read_data()
    print("Done")