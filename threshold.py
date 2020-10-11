import cv2
import numpy as np

imgFruit = cv2.imread('retina-mata.jpg')

grayFruit = cv2.cvtColor(imgFruit, cv2.COLOR_RGB2GRAY)

grayFruit_3_channel = cv2.cvtColor(grayFruit, cv2.COLOR_GRAY2BGR)

retval2,threshold2 = cv2.threshold(grayFruit_3_channel,125,225,cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.dilate(threshold2,kernel,iterations = 1)

image_show = np.hstack((imgFruit, grayFruit_3_channel))
cv2.imshow('Hasil Citra RGB Menjadi Citra Gray', image_show)
cv2.imshow('threshold',threshold2)
cv2.imshow('erosion', erosion)
cv2.waitKey()