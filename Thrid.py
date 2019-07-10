#!/usr/bin/python
import cv2
import numpy as np

background = None
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))
cap = cv2.VideoCapture('traffic.avi')

if (cap.isOpened() == False):
    print("Error opening vedio stream of file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        if background is None:
            background = gray_frame
            continue
        
        diff = cv2.absdiff(background, gray_frame)
        
        diff = cv2.morphologyEx(diff, cv2.MORPH_TOPHAT, kernel)

        diff = cv2.threshold(diff, 0,255, cv2.THRESH_OTSU)[1]
        
        diff = cv2.dilate(diff, es, iterations=2)

        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 1500:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(gray_frame, [box], 0, (0,255,0),2)
        
        cv2.imshow('contours', gray_frame)
        # cv2.imshow('dis', diff)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()

cv2.destroyAllWindows()