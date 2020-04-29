#!/usr/bin/env python

import cv2
import sys
import csv
import time
import numpy as np


def cropimg(tmask, test_hsv):
    tImage = tmask
    timgCrop = None
    center = []
    # Find only external contours
    contour, hierarchy = cv2.findContours(tImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by size
    contSort = sorted(contour, key=cv2.contourArea, reverse=True)[:1]

    # Check for Blank wall
    if hierarchy is not None:
        # Check if contour is big enough
        for n in range(len(contour)):
            M = cv2.moments(contour[n])
            if M['m00'] > 4000:
                x, y, w, h = cv2.boundingRect(contour[n])
                rect = cv2.minAreaRect(contour[n])
                w = int(rect[1][0])
                # h = int(rect[1][1])
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                y1 = int(y - h / 2)
                y2 = int(y + h / 2)
                x1 = int(x - w / 2)
                x2 = int(x + w / 2)

                # Create the cropped image
                timgCrop = test_hsv[y1:y2, x1:x2].copy()

                if not center:
                    center = [[x]]
                else:
                    center.append([x])

    # If the wall is blank then AKA found no contours large enough, then just send back the original image
    if len(center) is 0:
        timgCrop = test_hsv

    return timgCrop



### Load training images and labels

with open('./2019imgs/train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

# Read in images and blur
kernel = np.ones((5, 5), np.uint8)
images_bgr = [cv2.imread("./2019imgs/"+lines[i][0]+".png",1) for i in range(len(lines))]
blur = [cv2.GaussianBlur(images_bgr[i], (5, 5),0) for i in range(len(images_bgr))]
images_hsv = [cv2.cvtColor(blur[i], cv2.COLOR_BGR2HSV) for i in range(len(blur))]

# Ranges to just get any color vs white
lower = np.array([0,40,0])
upper = np.array([180,255,255])

#Generate Mask
mask = [cv2.inRange(images_hsv[i],lower, upper) for i in range(len(images_hsv))]

# Get rid of the arrow in the center
imgClose = [cv2.morphologyEx(mask[i], cv2.MORPH_CLOSE, kernel) for i in range(len(mask))]

# cv2.imshow("mask", imgClose[0])
# cv2.imshow("original", images_hsv[0])
# cv2.waitKey(0)

imagesCrop = []
for i in range(len(mask)):
    imgCrop = cropimg(imgClose[i], images_hsv[i])
    imagesCrop.append(imgCrop)

#Finalize formatting to feed to KNN to train
imagesFinal = np.array([cv2.resize(imagesCrop[i], (33, 25)) for i in range(len(imagesCrop))])
# train = [np.asarray(imagesFinal[i]) for i in range(len(imagesCrop))]
train = imagesFinal.flatten().reshape(len(lines), 33 * 25 * 3)
train_data = train.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

### Run test images
with open('./2019imgs/test.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

correct = 0.0
confusion_matrix = np.zeros((6,6))

for i in range(len(lines)):
    original_img = cv2.imread("./2019imgs/"+lines[i][0]+".png",1)
    # cv2.imshow("Original Image", original_img)
    test_img = cv2.cvtColor(cv2.imread("./2019imgs/"+lines[i][0]+".png",1), cv2.COLOR_BGR2HSV)

    # Generate Mask
    test_mask = cv2.inRange(test_img, lower, upper)

    # Clean up areas
    test_Close = cv2.morphologyEx(test_mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed image", test_Close)
    test_crop = cropimg(test_Close, test_img)

    test_final = np.array(cv2.resize(test_crop, (33, 25)))
    # cv2.imshow("Resized Image", test_final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_final = test_final.flatten().reshape(1, 33*25*3)
    test_final = test_final.astype(np.float32)

    test_label = np.int32(lines[i][1])

    ret, results, neighbours, dist = knn.findNearest(test_final, 3)

    if test_label == ret:
        print(lines[i][0], " Correct, ", ret)
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(lines[i][0], " Wrong, ", test_label, " classified as ", ret)
        print("\tneighbours: ", neighbours)
        print("\tdistances: ", dist)



print("\n\nTotal accuracy: ", correct/len(lines))
print(confusion_matrix)
