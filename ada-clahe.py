import numpy as np
import cv2 as cv

img = cv.imread('retina-mata.jpg',0)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl2 = clahe.apply(cl1)

kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(cl2,kernel,iterations = 1)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl3 = clahe.apply(erosion)

kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(cl3,kernel,iterations = 1)

ret,thresh1 = cv.threshold(dilation,80,255,cv.THRESH_BINARY_INV)

cv.imshow('original',img)
cv.imshow('clahe1',cl2)
cv.imshow('erosion',erosion)
cv.imshow('clahe 2',cl3)
cv.imshow('dilation',dilation)
cv.imshow('threshold',thresh1)
cv.waitKey()

