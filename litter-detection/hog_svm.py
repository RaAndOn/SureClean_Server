
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
        self.pos_dir = "/Users/AtulyaRavishankar/Documents/Dataset/pos_aug/" # Modify this if you have the data stored locally
        self.neg_dir = "/Users/AtulyaRavishankar/Documents/Dataset/neg_aug/" # Modify this if you have the data stored locally
        self.ignore_file = ".DS_Store"
        pos_files = os.listdir(self.pos_dir)
        neg_files = os.listdir(self.neg_dir)
        random.shuffle(pos_files)
        random.shuffle(neg_files)
        posIdx = int(len(pos_files) * self.split)
        negIdx = int(len(neg_files) * self.split)
        (self.pos_train, self.pos_test) = (pos_files[:posIdx], pos_files[posIdx:])
        (self.neg_train, self.neg_test) = (neg_files[:negIdx], neg_files[negIdx:]) 
        self.numTrainingImages = len(self.pos_train) + len(self.neg_train)
        self.numTestingImages = len(self.pos_test) + len(self.neg_test)
        self.img_dim = 70
        # HOG hyperparameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (3, 3)
        self.hog_block_norm = "L1"
        self.hog_visualize = False
        self.hog_transform_sqrt = False
        self.hog_feature_vector = True
        self.hog_size = 2916 #8100
        # SVM hyperparameters
        self.svm_coeff_file = "svm_coeffs.npy"
        self.svm_intercept_file = "svm_intercept.npy"
        self.svm_pkl_file = "svm_model.pkl"
    
    def get_training_data(self):
        print("Getting training data")
        idx = 0
        training_data = np.zeros((self.img_dim, self.img_dim, self.numTrainingImages))
        training_labels = np.zeros(self.numTrainingImages)
        print("    Fetching positive samples")
        for file in self.pos_train:
            if (file != self.ignore_file):
                fileName = self.pos_dir + file
                img = cv2.imread(fileName)
                if (img is not None):
                    training_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    training_labels[idx] = 1
                    idx += 1
        print("    Fetching negative samples")
        for file in self.neg_train:
            if (file != self.ignore_file):
                fileName = self.neg_dir + file
                img = cv2.imread(fileName)
                if (img is not None):
                    training_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    training_labels[idx] = 0
                    idx += 1
        return training_data.astype('float'), training_labels
    
    def get_testing_data(self):
        print("Getting test data")
        idx = 0
        testing_data = np.zeros((self.img_dim, self.img_dim, self.numTestingImages))
        testing_labels = np.zeros(self.numTestingImages)
        print("    Fetching positive samples")
        for file in self.pos_test:
            if (file != self.ignore_file):
                fileName = self.pos_dir + file
                img = cv2.imread(fileName)
                if (img is not None):
                    testing_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    testing_labels[idx] = 1
                    idx += 1
        print("    Fetching negative samples")
        for file in self.neg_test:
            if (file != self.ignore_file):
                fileName = self.neg_dir + file
                img = cv2.imread(fileName)
                if (img is not None):
                    testing_data[:, :, idx] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
                    testing_labels[idx] = 0
                    idx += 1
        return testing_data.astype('float'), testing_labels
    
    def get_feature(self, patch_H):
        feat = hog(patch_H, cells_per_block=self.hog_cells_per_block,
                        pixels_per_cell=self.hog_pixels_per_cell,
                        orientations=self.hog_orientations,
                        block_norm=self.hog_block_norm,
                        transform_sqrt=self.hog_transform_sqrt,
                        feature_vector=self.hog_feature_vector)
        return feat

    def train_svm(self):
        print("Training SVM")
        features = np.zeros((self.numTrainingImages, self.hog_size))
        data, labels = self.get_training_data() # (100, 100, 1200), (1200,)
        print("    Creating features")
        for i in range(self.numTrainingImages):
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
    
    def test_svm(self):
        print("Testing SVM")
        with open(self.svm_pkl_file, 'rb') as file:  
            clf = pickle.load(file)
        features = np.zeros((self.numTestingImages, self.hog_size))
        data, labels = self.get_testing_data()
        print("    Creating features.")
        for i in range(self.numTestingImages):
            img = data[:, :, i]
            feat = self.get_feature(img)
            features[i, :] = feat
        print("    Scoring")
        acc = clf.score(features.astype('float'), labels)
        print("Accuracy: %f" %acc)
        return acc
    
    def predict(self, patch, classifier):
        img = patch[:, :, 0]
        feat = self.get_feature(img)
        label_clf = classifier.predict(feat.reshape((1, -1)))
        return label_clf[0]

if __name__ == "__main__":
    random.seed(0)
    data = Dataset(split=0.95)
    data.train_svm()
    data.test_svm()
    
