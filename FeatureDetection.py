import cv2
import numpy as np
#picture should have adequate feature
#Getting the Image ready for feature detection
#ORB Algorithm
input_image = cv2.imread('BLUE_BOOK.jpg')
input_image = cv2.resize(input_image, (450,500),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints with ORB
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# draw only the location of the keypoints without size or
final_keypoints = cv2.drawKeypoints(gray_image, keypoints,input_image,(0,255,0))

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
