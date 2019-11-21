import os
import numpy as np
import cv2


if __name__ == "__main__":
    idx = 1
    file_names = os.listdir("images/input/")
    results = dict()
    idx = 0
    for file in file_names:
        if (file != ".DS_Store" and file != ".gitignore"):
            debug_file = file[:-4]
            print("Processing file %d of %d" %(idx, len(file_names)-2))
            fileName = "images/input/" + file
            img = cv2.imread(fileName)
            img = cv2.resize(img, (1280, 720))
            cv2.imwrite("tools/" + debug_file + "_lowres.jpg", img)
        idx += 1
    print("Done")