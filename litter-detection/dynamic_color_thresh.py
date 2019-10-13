"""
Takes in images from the images/input/ directory, finds the litter objects in the images
and saves the output to the images/output/ directory. A JSON file containing the results
is also generated.

Author: Atulya Ravishankar
Updated: 09/08/2019
"""

import pickle
import os
import copy
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from hog_svm import Dataset

debug_thresh = True
debug_file = ""

data = Dataset(split=0.95)

# OpenCV range for HSV channels:
# H: [0 180]
# S: [0 255]
# V: [0 255]

class LitterDetector(object):

    def __init__(self):
        self.h_thresh_width = 4.5
        self.s_thresh_width = 4.5
        self.margin = 5
        self.blob_lower_size = 5 #20
        self.blob_upper_size = 150 #150
        self.patch_size = 70 # 70
        self.img_width = 1280  #4000
        self.img_height = 720 #2250
        self.svm_coeffs = np.load("model/grass/grass_svm_coeffs.npy")
        self.svm_intercept = np.load("model/grass/grass_svm_intercept.npy")
        self.svm_pkl_file = "model/grass/grass_svm_model.pkl"
        with open(self.svm_pkl_file, 'rb') as file:  
            self.clf = pickle.load(file)
        if (self.clf is None):
            raise Exception("Failed to load SVM model.")
    
    '''
    Reads in an image 'fileName' and outputs the image in BGR, RGB and HSV color spaces.
    '''
    def read_image(self, fileName):
        img_bgr = cv2.imread(fileName)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return (img_bgr, img_rgb, img_hsv)
    
    '''
    Computes dynamic thresholds in the HSV space for the image 'img_hsv'.
    '''
    def compute_thresholds(self, img_hsv):
        [H, S, _] = [img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]]
        thresh = dict()
        H_mean = np.mean(H)
        H_std = np.std(H)
        S_mean = np.mean(S)
        S_std = np.std(S)
        thresh["H_min"] = H_mean - self.h_thresh_width*H_std
        thresh["H_max"] = H_mean + self.h_thresh_width*H_std
        thresh["S_min"] = S_mean - self.s_thresh_width*S_std
        thresh["S_max"] = 255
        thresh["V_min"] = 0
        thresh["V_max"] = 255
        return thresh
    
    '''
    Compute the centroid of a contour 'contour'.
    '''
    def get_contour_centroid(self, contour):
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = 0
            cY = 0
        return (cX, cY)
    
    '''
    Extracts a patch from the image centered at the centroid of 'ctr'.
    '''
    def get_patch_idx(self, ctr):
        (cX, cY) = self.get_contour_centroid(ctr)
        halfPatch = int(self.patch_size/2)
        if (cX < halfPatch):
            minX = 0
            maxX = minX + self.patch_size - 1
        elif (cX > self.img_width-halfPatch):
            maxX = self.img_width - 1
            minX = maxX - self.patch_size + 1
        else:
            minX = cX - halfPatch
            maxX = minX + self.patch_size - 1
        if (cY < halfPatch):
            minY = 0
            maxY = minY + self.patch_size - 1
        elif (cY > self.img_height-halfPatch):
            maxY = self.img_height - 1
            minY = maxY - self.patch_size + 1
        else:
            minY = cY - halfPatch
            maxY = minY + self.patch_size - 1
        return (minX, maxX, minY, maxY)
    
    '''
    Classifies an image patch 'patch' as litter or not-litter.
    '''
    def predict(self, patch):
        global data
        img = patch[:, :, 0]
        feat = data.get_feature(img)
        label_clf = self.clf.predict(feat.reshape((1, -1)))
        return label_clf[0]
    
    '''
    Determines if a contour 'ctr' is valid (i.e. should be classified).
    '''
    def is_valid_contour(self, img_hsv, ctr):
        global data
        size_of_blob = ctr.shape[0]
        (cX, cY) = self.get_contour_centroid(ctr)
        within_bounds = (cX >= self.margin) and (cX <= (self.img_width-self.margin)) and (cY >= self.margin) and (cY <= (self.img_height-self.margin))
        if (within_bounds and size_of_blob >= self.blob_lower_size and size_of_blob <= self.blob_upper_size):
            (x1, x2, y1, y2) = self.get_patch_idx(ctr)
            patch = img_hsv[y1:y2+1, x1:x2+1, :]
            return True#(self.predict(patch) > 0.5)
        return False

    '''
    Thresholds the image 'img_hsv' based on the thresholds stored in 'thresh'.
    '''
    def threshold_image(self, img_hsv, thresh):
        global debug_thresh, debug_file
        img_thresh = 255 - cv2.inRange(img_hsv, (thresh["H_min"], thresh["S_min"], thresh["V_min"]), (thresh["H_max"], thresh["S_max"], thresh["V_max"]))
        _, contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE) 
        # mask = np.ones(img_thresh.shape[:2], dtype="uint8") * 255
        valid_contours = []
        for c in contours:
            if self.is_valid_contour(img_hsv, c):
                valid_contours.append(c)
            # else:
            #     cv2.drawContours(mask, [c], -1, 0, -1)
        # image = cv2.bitwise_and(img_thresh, img_thresh, mask=mask).astype('uint8')
        # output = (255*image).astype("uint8")
        if (debug_thresh):
            cv2.imwrite("images/thresh/"+ debug_file + "_thresh.jpg", img_thresh)
        output = img_thresh
        return (output, valid_contours)
    
    '''
    Plots the locations of litter ('contours') identified in 'image'.
    '''
    def visualize_litter_locations(self, image, points):
        for [cX, cY] in points:
            cv2.circle(image, (cX, cY), 18, (0, 0, 255), 3, 8, 0)
        return image

'''
Takes in a list of points (x, y) that returns a subset of that list such that no
two points are within 5px of each other.
'''
def filter_points(contours):
    filtered = []
    to_remove = [False for _ in contours]
    for i in range(len(contours)):
        if not to_remove[i]:
            for j in range(len(contours)):
                if (i != j):
                    (cXi, cYi) = ld.get_contour_centroid(contours[i])
                    (cXj, cYj) = ld.get_contour_centroid(contours[j])
                    dist = np.linalg.norm(np.array([cXi, cYi]) - np.array([cXj, cYj]))
                    if (dist < 10):
                        to_remove[j] = True
            filtered.append(contours[i])
    final_positions = []
    for ctr in filtered:
        final_positions.append(ld.get_contour_centroid(ctr))
    return final_positions

'''
Uses the litter detector results, filters them and saves it to a JSON file.
'''
def filter_and_save_results(results):
    with open('results.json', 'w') as results_file:
        json.dump(results, results_file)


if __name__ == "__main__":
    idx = 1
    fileNames = os.listdir("images/input/")
    ld = LitterDetector()
    results = dict()
    for file in fileNames:
        if (file != ".DS_Store" and file != ".gitignore"):
            debug_file = file[:-4]
            print("Processing file %d of %d" %(idx, len(fileNames)-2))
            fileName = "images/input/" + file
            (img_bgr, img_rgb, img_hsv) = ld.read_image(fileName)
            thresh = ld.compute_thresholds(img_hsv)
            (img_bin, contours) = ld.threshold_image(img_hsv, thresh)
            # Filter points to remove duplicates
            results[file] = filter_points(contours)
            output = ld.visualize_litter_locations(img_bgr, results[file])
            cv2.imwrite("images/output/"+file[:-4]+"_result.jpg", output)
            idx += 1
    filter_and_save_results(results)
    print("Done")
