#!/usr/bin/python
import cv2
import numpy as np

#打开图片
img=cv2.imread('rice.jpg',0)

#构造模板，5次腐蚀，5次膨胀，得到背景
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(img,kernel,iterations=5)
dilation=cv2.dilate(erosion,kernel,iterations=5)

#原图减去背景得到米粒形状
backImg=dilation
rice=img-backImg

#OSTU二值化
th1,ret1=cv2.threshold(rice,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#轮廓检测
contours,hierarchy=cv2.findContours(ret1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

areas = []
lenth = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 50:
        areas.append(area)
        rec = cv2.minAreaRect(cnt)
        lenth.append(rec[1][1])
        print("面积: %f 长: %f" % (area, rec[1][1]) )

print("ave Area: %d ave Len %d" % (sum(areas)/len(areas), sum(lenth)/len(lenth)))


'''
#遍历得到最大面积的米粒
maxC=-1
maxS=-1
for cnt in contours:
    tempS=cv2.contourArea(cnt)
    if maxS<tempS:
        maxS=tempS
        maxC=tempC=cv2.arcLength(cnt,True)
        contour=cnt

#在img中画出最大面积米粒
cv2.drawContours(img,[contour],-1,(0,0,255,),1)

cv2.imshow('image',rice)
print('面积最大：',maxS)
print('对应米粒周长：',maxC)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''