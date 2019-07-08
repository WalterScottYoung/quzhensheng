#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))

imgobj = cv2.imread("rice.jpg")

'''
cv2.namedWindow("imgobj") #创建窗口并显示的是图像类型
cv2.imshow("imgobj",imgobj)
cv2.waitKey(3000)        #等待事件触发，参数0表示永久等待
cv2.destroyAllWindows()   #释放窗口

tophat = cv2.morphologyEx(imgobj, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("tophat",tophat)
cv2.waitKey(3000)        #等待事件触发，参数0表示永久等待
cv2.destroyAllWindows()   #释放窗口
'''
imgobj = cv2.morphologyEx(imgobj, cv2.MORPH_TOPHAT, kernel)
imgobj = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY)

ret1,th1 = cv2.threshold(imgobj,127,255, cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(imgobj,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(imgobj ,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the imgobjs and their histograms
images = [imgobj, 0, th1,
          imgobj, 0, th2,
          imgobj, 0, th2-10/255,
          blur, 0, th3-10/255] 
titles = ['Original Noisy imgobj','Histogram','Global Thresholding (v=127)',
          'Original Noisy imgobj','Histogram',"Otsu's Thresholding",
          'Original Noisy imgobj th reduced','Histogram',"Otsu's Thresholding th reduced",
          'Gaussian filtered imgobj','Histogram',"Otsu's Thresholding"]

for i in range(4):
    plt.subplot(4,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

th_floodfill = th2.copy()
h, w = th2.shape[:2]
mask = np.zeros((h+2, w+2), np.unit8)

cv2.floodFill(th_floodfill, mask, (0,0), 255)

th_floodfill_inv = cv2.bitwise_not(th_floodfill)

im_out = th2 | th_floodfill_inv

cv2.imshow("Thresholded Image", th2)
cv2.imshow("Floodfilled Image", th_floodfill)
cv2.imshow("Inverted Floodfilled Image", th_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)


 
''' 
 hist1 = cv2.calcHist([imgobj], [0], None, [256], [0.0, 255.0])
 pyplot.plot(range(256), hist1, 'r')
 pyplot.show()
 cv2.imshow('img1',imgobj)
 cv2.waitKey(3000)
'''