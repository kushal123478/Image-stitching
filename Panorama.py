# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:08:11 2019

@author: KUSHAL
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio



def panorama(img,img_,name):
    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    orb = cv2.ORB_create()
    
    
    # find key points
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    good = sorted(matches, key = lambda x:x.distance)
    draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=None,
                           flags=2)


    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.2)
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        print("Displacement of corners between images:")
        print(dst-pts)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))
    dst = cv2.warpPerspective(img_,M,(img.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0],0:img.shape[1]] = img

    def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop top
        if not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop top
        if not np.sum(frame[:,0]):
            return trim(frame[:,1:])
        #crop top
        if not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])
        return frame
    plt.figure(figsize=(20,10))
    plt.imshow(trim(dst))
    imageio.imwrite("result"+name+".png", trim(dst))
    return trim(dst)

name = '\Goodwin'
img0 = imageio.imread(r'C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 3\ECE5554 FA19 HW3 images' + name + '0.png')
img1 = imageio.imread(r'C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 3\ECE5554 FA19 HW3 images' + name + '1.png')
img2 = imageio.imread(r'C:\Users\Kushal Patel\Desktop\Courses\Computer Vision\Homework 3\ECE5554 FA19 HW3 images' + name + '2.png')

inter1 = panorama(img0, img1, "_Goodwin")
final = panorama(inter1, img2, "_Goodwin")