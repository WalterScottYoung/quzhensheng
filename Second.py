#!/usr/bin/python
import cv2
import numpy as np

# Open image
imgOriginal=cv2.imread('stuff.jpg')

# transform image to gray
img=cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
img = (255-img)

# change lights
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25, 25))
nextkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

img=cv2.dilate(img,nextkernel,iterations=5)

# OTSU
res, dst = cv2.threshold(img,0, 255,cv2.THRESH_OTSU)

# contours detection
contours,hierarchy=cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(dst, contours, -1, (120, 0,0),2)

ares = -1
maxcont = []
for cont in contours:
    currentares = cv2.contourArea(cont)
    if currentares > ares:
        ares = currentares
        maxcont = cont

# draw the object
rect = cv2.minAreaRect(maxcont)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(imgOriginal,[box],0,(0,0, 
255),2)

cv2.namedWindow("Original", 2)
cv2.imshow("Original", imgOriginal)

cv2.namedWindow("GREY", 2)
cv2.imshow("GREY", img)

cv2.namedWindow("Processed", 2)
cv2.imshow("Processed", dst)

cv2.waitKey()