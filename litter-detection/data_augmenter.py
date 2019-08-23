"""
Takes the relatively small set of training data images and augments them
to expand the total training set. It does so by flipping and rotating 
each of the original images. This helps the training set be more representative
of geometric variations of the data in the real world (e.g. arbitrary rotations).

Author: Atulya Ravishankar
Updated: 04/17/2019
"""

import os
import numpy as np
import cv2

img_size = 70
num_angles = 36

ctr = 0

'''
Rotates the image 'img' in increments of 360/'num_angles'.
'''
def augment_image_rot(img):
    global img_size, num_angles
    (rows, cols, _) = img.shape
    images = np.zeros((img_size, img_size, 3, num_angles))
    angles = np.linspace(0, 350, num_angles) # 0, 10, 20, ..., 330, 340, 350
    start_idx = int((100-img_size) / 2)
    end_idx = start_idx + img_size
    for i in range(len(angles)):
        angle = angles[i]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        dst_cropped = dst[start_idx:end_idx, start_idx:end_idx, :]
        images[:, :, :, i] = dst_cropped
    return images
    
'''
Augments a single image 'img'.
Does so by reflecting and rotating the image.
'''
def augment_image(img):
    global img_size, num_angles
    flip_none = img
    flip_ud = cv2.flip(img, 0)
    flip_lr = cv2.flip(img, 1)
    flip_udlr = cv2.flip(flip_ud, 1)
    aug_images = np.zeros((img_size, img_size, 3, 4*num_angles))
    aug_images[:, :, :, :num_angles] = augment_image_rot(flip_none)
    aug_images[:, :, :, num_angles:2*num_angles] = augment_image_rot(flip_ud)
    aug_images[:, :, :, 2*num_angles:3*num_angles] = augment_image_rot(flip_lr)
    aug_images[:, :, :, 3*num_angles:4*num_angles] = augment_image_rot(flip_udlr)
    return aug_images

'''
Augments all the images in 'directory_in_name' and saves the output to 'directory_out_name'.
Currently augments by rotating and reflecting images.
'''
def augment_dir(directory_in_name, directory_out_name):
    global ctr
    directory = os.listdir(directory_in_name)
    for file in directory:
        file_name = directory_in_name + file
        if (file_name != ".DS_Store"):
            img = cv2.imread(file_name)
            if (img is not None):
                aug_images = augment_image(img)
                num_images = aug_images.shape[3]
                for i in range(num_images):
                    cur_img = aug_images[:, :, :, i]
                    cv2.imwrite(directory_out_name + str(ctr) + "_bootstrapped_3.jpg", cur_img)
                    ctr += 1

if __name__ == "__main__":
    # Modify the file paths below if you have the data stored locally
    # Note that the bootstrap/ directory does not exist in the repository, it is just a placeholder path
    # Contact Atulya to get the final dataset.
    augment_dir("bootstrap/pos/", "/Users/AtulyaRavishankar/Documents/Dataset/pos_aug/")
    augment_dir("bootstrap/neg/", "/Users/AtulyaRavishankar/Documents/Dataset/neg_aug/")


