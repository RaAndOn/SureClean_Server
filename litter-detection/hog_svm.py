"""
Manager SVM classifier
- Training: uses training data and saves the model to disk.
- Testing/Predicting: uses saved model to classify test image(s)

Author: Atulya Ravishankar
Updated: 09/08/2019
"""

import pickle
import os
import random
import math
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report,accuracy_score
from matplotlib import pyplot as plt

# NOTE: The dataset needed to train the SVM is not in the Git repository due to its immense size.
#       Contact Atulya if you would like the dataset.

class Dataset(object):

    def __init__(self, split=0.75):
        # Data hyperparameters
        self.split = split
        self.pos_dir = "dataset/grass/augmented/pos/"
        self.neg_dir = "dataset/grass/augmented/neg/"
        self.ignore_file = ".DS_Store"
        pos_files = os.listdir(self.pos_dir)
        neg_files = os.listdir(self.neg_dir)
        random.shuffle(pos_files)
        random.shuffle(neg_files)
        pos_idx = int(len(pos_files) * self.split)
        neg_idx = int(len(neg_files) * self.split)
        (self.pos_train, self.pos_test) = (pos_files[:pos_idx], pos_files[pos_idx:])
        (self.neg_train, self.neg_test) = (neg_files[:neg_idx], neg_files[neg_idx:]) 
        self.num_training_images = len(self.pos_train) + len(self.neg_train)
        self.num_testing_images = len(self.pos_test) + len(self.neg_test)
        self.img_dim = 70
        # HOG hyperparameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (3, 3)
        self.hog_block_norm = "L1"
        self.hog_visualize = False
        self.hog_transform_sqrt = False
        self.hog_feature_vector = True
        self.hog_size = 2916
        # SVM hyperparameters
        self.svm_coeff_file = "model/grass/grass_svm_coeffs.npy"
        self.svm_intercept_file = "model/grass/grass_svm_intercept.npy"
        self.svm_pkl_file = "model/grass/grass_svm_model.pkl"
    
    '''
    Loads training data (images and labels) for classifier.
    '''
    def get_training_data(self):
        print("Getting training data")
        idx = 0
        training_data = np.zeros((self.img_dim, self.img_dim, self.num_training_images))
        training_labels = np.zeros(self.num_training_images)
        print("    Fetching positive samples")
        for file in self.pos_train:
            if (file != self.ignore_file):
                file_name = self.pos_dir + file
                img = cv2.imread(file_name)
                if (img is not None):
                    training_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    training_labels[idx] = 1
                    idx += 1
        print("    Fetching negative samples")
        for file in self.neg_train:
            if (file != self.ignore_file):
                file_name = self.neg_dir + file
                img = cv2.imread(file_name)
                if (img is not None):
                    training_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    training_labels[idx] = 0
                    idx += 1
        return training_data.astype('float'), training_labels
    
    '''
    Loads testing data (images and labels) for classifier.
    '''
    def get_testing_data(self):
        print("Getting test data")
        idx = 0
        testing_data = np.zeros((self.img_dim, self.img_dim, self.num_testing_images))
        testing_labels = np.zeros(self.num_testing_images)
        print("    Fetching positive samples")
        for file in self.pos_test:
            if (file != self.ignore_file):
                file_name = self.pos_dir + file
                img = cv2.imread(file_name)
                if (img is not None):
                    testing_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    testing_labels[idx] = 1
                    idx += 1
        print("    Fetching negative samples")
        for file in self.neg_test:
            if (file != self.ignore_file):
                file_name = self.neg_dir + file
                img = cv2.imread(file_name)
                if (img is not None):
                    testing_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    testing_labels[idx] = 0
                    idx += 1
        return testing_data.astype('float'), testing_labels
    
    '''
    Converts an image patch into a HoG feature for the classifier.
    '''
    def get_feature(self, patch_H):
        feat = hog(patch_H, cells_per_block=self.hog_cells_per_block,
                        pixels_per_cell=self.hog_pixels_per_cell,
                        orientations=self.hog_orientations,
                        block_norm=self.hog_block_norm,
                        transform_sqrt=self.hog_transform_sqrt,
                        feature_vector=self.hog_feature_vector)
        return feat

    '''
    Trains a Support Vector Machine (SVM) classifier and saves the model for future use.
    '''
    def train_svm(self):
        print("Training SVM")
        features = np.zeros((self.num_training_images, self.hog_size))
        data, labels = self.get_training_data() # (100, 100, 1200), (1200,)
        print("    Creating features")
        for i in range(self.num_training_images):
            img = data[:, :, i]
            feat = self.get_feature(img)
            features[i, :] = feat
        print("    Computing weights")
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(features.astype('float'), labels)
        np.save(self.svm_coeff_file, clf.coef_)
        np.save(self.svm_intercept_file, clf.intercept_)
        with open(self.svm_pkl_file, 'wb') as file:  
            pickle.dump(clf, file)
        return clf
    
    '''ical

    Tests a Support Vector Machine (SVM) model.
    Model is loaded from disk.
    '''
    def test_svm(self):
        print("Testing SVM")
        with open(self.svm_pkl_file, 'rb') as file:  
            clf = pickle.load(file)
        features = np.zeros((self.num_testing_images, self.hog_size))
        data, labels = self.get_testing_data()
        print("    Creating features.")
        for i in range(self.num_testing_images):
            img = data[:, :, i]
            feat = self.get_feature(img)
            features[i, :] = feat
        print("    Scoring")
        acc = clf.score(features.astype('float'), labels)
        print("Accuracy: %f" %acc)
        return acc
    
    '''
    Classifies an image patch using the classifier.
    '''
    def predict(self, patch, classifier):
        img = patch[:, :, 0]
        feat = self.get_feature(img)
        label_clf = classifier.predict(feat.reshape((1, -1)))
        return label_clf[0]


if __name__ == "__main__":
    data = Dataset(split=0.97)
    data.train_svm()
    data.test_svm()
    
