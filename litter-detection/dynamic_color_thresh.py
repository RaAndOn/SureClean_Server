"""
Takes in images from the svd_images/input/ directory, finds the litter objects in the images
and saves the output to the svd_images/output/ directory.
"""

import pickle
import random
import os
import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

from hog_svm import Dataset

random.seed(0)
data = Dataset(split=0.95)

# OpenCV range for HSV channels:
# H: [0 180]
# S: [0 255]
# V: [0 255]

class LitterDetector(object):

    def __init__(self):
        self.h_thresh_width = 4.5
        self.s_thresh_width = 4.5
        self.blob_lower_size = 20
        self.blob_upper_size = 150
        self.patch_size = 70
        self.img_width = 4000
        self.img_height = 2250
        self.svm_coeffs = np.load("svm_coeffs.npy")
        self.svm_intercept = np.load("svm_intercept.npy")
        self.svm_pkl_file = "svm_model.pkl"
        with open(self.svm_pkl_file, 'rb') as file:  
            self.clf = pickle.load(file)
        if (self.clf is None):
            raise Exception("Failed to load SVM model.")
    
    def read_image_hsv(self, fileName):
        img_bgr = cv2.imread(fileName)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return (img_bgr, img_rgb, img_hsv)
    
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
    
    def get_contour_centroid(self, contour):
        M = cv2.moments(contour)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = 0
            cY = 0
        return (cX, cY)
    
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
    
    def predict(self, patch):
        global data
        img = patch[:, :, 0]
        feat = data.get_feature(img)
        label_clf = self.clf.predict(feat.reshape((1, -1)))
        return label_clf[0]
    
    def is_valid_contour(self, img_hsv, ctr):
        global data
        size_of_blob = ctr.shape[0]
        if (size_of_blob >= self.blob_lower_size and size_of_blob <= self.blob_upper_size):
            (x1, x2, y1, y2) = self.get_patch_idx(ctr)
            patch = img_hsv[y1:y2+1, x1:x2+1, :]
            return (self.predict(patch) > 0.5)
        return False

    def threshold_image(self, img_hsv, thresh):
        img_thresh = 255 - cv2.inRange(img_hsv, (thresh["H_min"], thresh["S_min"], thresh["V_min"]), (thresh["H_max"], thresh["S_max"], thresh["V_max"]))
        _, contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) 
        mask = np.ones(img_thresh.shape[:2], dtype="uint8") * 255
        valid_contours = []
        for c in contours:
            if self.is_valid_contour(img_hsv, c):
                valid_contours.append(c)
            else:
                cv2.drawContours(mask, [c], -1, 0, -1)
        image = cv2.bitwise_and(img_thresh, img_thresh, mask=mask).astype('uint8')
        output = (255*image).astype("uint8")
        return (output, valid_contours)
    
    def visualize_litter_locations(self, image, contours):
        centroids = []
        for c in contours:
            (cX, cY) = self.get_contour_centroid(c)
            centroids.append((cX, cY))
            cv2.circle(image, (cX, cY), 50, (0, 0, 255), 3, 8, 0)
        return image


if __name__ == "__main__":
    idx = 1
    fileNames = os.listdir("svd_images/input/")
    ld = LitterDetector()
    for file in fileNames:
        if (file != ".DS_Store"):
            print("Processing file %d of %d" %(idx, len(fileNames)-1))
            fileName = "svd_images/input/" + file
            (img_bgr, img_rgb, img_hsv) = ld.read_image_hsv(fileName)
            thresh = ld.compute_thresholds(img_hsv)
            (img_bin, contours) = ld.threshold_image(img_hsv, thresh)
            output = ld.visualize_litter_locations(img_bgr, contours)
            cv2.imwrite("svd_images/output/"+file[-4:]+"_result.jpg", output)
            idx += 1
    print("Done")
