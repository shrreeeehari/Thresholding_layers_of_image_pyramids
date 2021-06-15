import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/shrih/OneDrive/Pictures/test1/image.orig/36.jpg')
level = img.copy()
gp_list = [level]

for i in range(4):
    level = cv2.pyrDown(level)
    gp_list.append(level)
    img_gray = cv2.cvtColor(level, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray,5)
    level1=cv2.medianBlur(level,1).astype('uint8')
    ret,th1 = cv2.threshold(level1,80,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = [ 'Gauss Py.-Global Thresholding','Adaptive Thresholding']
    images = [ th1,th2]
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
        
lp_list = [level]
for i in range(4, 0, -1):
    gaussian_extended = cv2.pyrUp(gp_list[i])
    laplacian = cv2.subtract(gp_list[i-1], gaussian_extended)
    img_gray = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray,5)
    level1=cv2.medianBlur(laplacian,1).astype('uint8')
    ret,th1 = cv2.threshold(level1,80,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img_blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = [ 'Laplacian Py.-Global Thresholding','Adaptive Thresholding']
    images = [ th1,th2]
    
    
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    

