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

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import clasifier_dl
from hog_svm import Dataset

debug_thresh = False
debug_file = ""

data = Dataset(split=0.95)

# OpenCV range for HSV channels:
# H: [0 180]
# S: [0 255]
# V: [0 255]

class LitterDetector(object):

    def __init__(self, path="", use_svm=True):
        self.use_svm = use_svm
        self.h_thresh_width = 4.5
        self.s_thresh_width = 4.5
        self.blob_lower_size = 5
        self.blob_upper_size = 150
        self.patch_size = 30
        self.img_width = 1280 
        self.img_height = 720
        self.svm_coeffs = np.load(path + "model/grass/grass_svm_coeffs_lowres.npy")
        self.svm_intercept = np.load(path + "model/grass/grass_svm_intercept_lowres.npy")
        self.svm_pkl_file = path + "model/grass/grass_svm_model_lowres.pkl"
        with open(self.svm_pkl_file, 'rb') as file:
            self.clf = pickle.load(file)
        if (self.clf is None):
            raise Exception("Failed to load SVM model.")
        self.dl_model_file = "checkpoint_AWS"
        checkpoint = torch.load(self.dl_model_file, map_location=torch.device("cpu"))
        self.net = checkpoint['network']
        self.net.load_state_dict(checkpoint['network_state_dict'])

    
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
        if (size_of_blob >= self.blob_lower_size and size_of_blob <= self.blob_upper_size):
            (x1, x2, y1, y2) = self.get_patch_idx(ctr)
            patch = img_hsv[y1:y2+1, x1:x2+1, :]
            return (self.predict(patch) > 0.5)
        return False

    '''
    Thresholds the image 'img_hsv' based on the thresholds stored in 'thresh'.
    '''
    def threshold_image(self, img_hsv, thresh):
        global debug_thresh, debug_file
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
        if (debug_thresh):
            cv2.imwrite("images/thresh/"+ debug_file + "_thresh.jpg", img_thresh)
        return (output, valid_contours)
    
    '''
    Plots the locations of litter ('contours') identified in 'image'.
    '''
    def visualize_litter_locations(self, image, points):
        for [cX, cY] in points:
            cv2.circle(image, (cX, cY), 50, (0, 0, 255), 3, 8, 0)
        return image

    def sliding_window(self, img_rgb):
        litter_locs = []
        stride = 20
        [H, W, _] = img_rgb.shape
        for startRow in range(0, H, stride):
            for startCol in range(0, W, stride):
                patch = img_rgb[startRow:startRow+self.patch_size, startCol:startCol+self.patch_size, :]
                is_litter = clasifier_dl.classify_image(self.net, patch)
                if (is_litter):
                    litter_locs.append((startRow+int(stride/2), startCol+int(stride/2)))
        return litter_locs

'''
Takes in a list of points (x, y) that returns a subset of that list such that no
two points are within 5px of each other.
'''
def filter_points(contours, ld):
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


def filter_sliding_window(locs):
    # Note: locs is [row, col] where we want to output [x, y]
    filtered = []
    to_remove = [False for _ in locs]
    for i in range(len(locs)):
        if not to_remove[i]:
            for j in range(len(locs)):
                if (i != j):
                    [rowi, coli] = locs[i]
                    [rowj, colj] = locs[j]
                    dist = np.linalg.norm(np.array([rowi, coli]) - np.array([rowj, colj]))
                    if (dist < 30):
                        to_remove[j] = True
            filtered.append([coli, rowi]) # switch order to make it x, y
    return filtered

'''
Uses the litter detector results, filters them and saves it to a JSON file.
'''
def filter_and_save_results(results):
    with open('results.json', 'w') as results_file:
        json.dump(results, results_file)


'''
This function is externally accessible to be run by the litter detection pipeline
'''
def run_dynamic_color_thresh(file_path, sureclean_server_path): 
    use_svm = True
    file = file_path.split('/')[-1].split('.')[0]
    ld = LitterDetector(path=sureclean_server_path+'/litter-detection/', use_svm=use_svm)
    results = dict()
    (img_bgr, img_rgb, img_hsv) = ld.read_image(file_path)
    if (use_svm):
        thresh = ld.compute_thresholds(img_hsv)
        (img_bin, contours) = ld.threshold_image(img_hsv, thresh)
        # Filter points to remove duplicates
        results[file] = filter_points(contours,ld)
    else:
        unfiltered_litter_locs = ld.sliding_window(img_rgb)
        results[file] = filter_sliding_window(unfiltered_litter_locs)
    output = ld.visualize_litter_locations(img_bgr, results[file])
    cv2.imwrite(sureclean_server_path+'/litter-detection/'+"images/output/"+file+"_result.jpg", output)
    return results


# if __name__ == "__main__":
#     idx = 1
#     file_names = os.listdir("images/input/")
#     ld = LitterDetector()
#     results = dict()
#     for file in file_names:
#         if (file != ".DS_Store" and file != ".gitignore"):
#             debug_file = file[:-4]
#             print("Processing file %d of %d" %(idx, len(fileNames)-2))
#             fileName = "images/input/" + file
#             (img_bgr, img_rgb, img_hsv) = ld.read_image(fileName)
#             thresh = ld.compute_thresholds(img_hsv)
#             (img_bin, contours) = ld.threshold_image(img_hsv, thresh)
#             # Filter points to remove duplicates
#             results[file] = filter_points(contours,ld)
#             output = ld.visualize_litter_locations(img_bgr, results[file])
#             cv2.imwrite("images/output/"+file[:-4]+"_result.jpg", output)
#             idx += 1
#     filter_and_save_results(results)
#     print("Done")
