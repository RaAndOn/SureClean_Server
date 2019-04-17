
import os
import numpy as np
import cv2

img_size = 70
numAngles = 36

ctr = 0

def augment_image_rot(img):
    global img_size, numAngles
    (rows, cols, _) = img.shape
    images = np.zeros((img_size, img_size, 3, numAngles))
    angles = np.linspace(0, 350, numAngles) # 0, 10, 20, ..., 330, 340, 350
    startIdx = int((100-img_size) / 2)
    endIdx = startIdx + img_size
    for i in range(len(angles)):
        angle = angles[i]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        dst_cropped = dst[startIdx:endIdx, startIdx:endIdx, :]
        images[:, :, :, i] = dst_cropped
    return images
    
def augment_image(img):
    global img_size, numAngles
    flip_none = img
    flip_ud = cv2.flip(img, 0)
    flip_lr = cv2.flip(img, 1)
    flip_udlr = cv2.flip(flip_ud, 1)
    aug_images = np.zeros((img_size, img_size, 3, 4*numAngles))
    aug_images[:, :, :, :numAngles] = augment_image_rot(flip_none)
    aug_images[:, :, :, numAngles:2*numAngles] = augment_image_rot(flip_ud)
    aug_images[:, :, :, 2*numAngles:3*numAngles] = augment_image_rot(flip_lr)
    aug_images[:, :, :, 3*numAngles:4*numAngles] = augment_image_rot(flip_udlr)
    return aug_images

def augment_dir(directory_in_name, directory_out_name):
    global ctr
    directory = os.listdir(directory_in_name)
    for file in directory:
        fileName = directory_in_name + file
        if (fileName != ".DS_Store"):
            img = cv2.imread(fileName)
            if (img is not None):
                aug_images = augment_image(img)
                numImages = aug_images.shape[3]
                for i in range(numImages):
                    cur_img = aug_images[:, :, :, i]
                    cv2.imwrite(directory_out_name + str(ctr) + ".jpg", cur_img)
                    ctr += 1

if __name__ == "__main__":
    # Modify the file paths below if you have the data stored locally
    # Note that the data/ directory does not exist in the repository, it is just a placeholder path
    # Contact Atulya to get the final dataset.
    augment_dir("data/pos/", "/Users/AtulyaRavishankar/Documents/Dataset/pos_aug/")
    augment_dir("data/neg/", "/Users/AtulyaRavishankar/Documents/Dataset/neg_aug/")


