# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:44:12 2022

@author: paulg
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import scipy.signal as sg
import cv2

import harris

img1 = Image.open("set1-1.png")
img1 = np.array(ImageOps.grayscale(img1))
img2 = Image.open("set1-2.png")
img2 = np.array(ImageOps.grayscale(img2))

R1,x1,y1=harris.harris(img1,seuil=0.2,taille_fenetre=3,sigma=2)

R2,x2,y2=harris.harris(img2,seuil=0.2,taille_fenetre=3,sigma=2)


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Points d'interets")
ax1.plot(y1,x1,'+r')
ax2.plot(y2,x2,'+r')

ax1.imshow(img1,cmap='gray')
ax2.imshow(img2,cmap='gray')

fig2, ax3 = plt.subplots(1, 1)

IMG=np.concatenate((img1,img2),axis=1)
ax3.plot(y1,x1,'+r')
ax3.plot(y2+img1.shape[1],x2,'+r')
ax3.imshow(IMG,cmap='gray')

patch=5
h,l=img1.shape

paireZMSDD=np.zeros((len(x1),len(x2)))

for i in range (len(x1)):
    for j in range (len(x2)):
        ZMSDD=0
        if (((x1[i]-patch)>=0)&((x1[i]+patch)<h)&((y1[i]-patch)>=0)&((y1[i]+patch)<l)):
            mean1=np.mean(img1[x1[i]-patch:x1[i]+patch+1,y1[i]-patch:y1[i]+patch+1])
            mean2=np.mean(img2[x2[j]-patch:x2[j]+patch+1,y2[j]-patch:y2[j]+patch+1])
            for x in range(-patch,patch+1):
                for y in range(-patch,patch+1):
                    ZMSDD+=(img1[x1[i]+x,y1[i]+y]-mean1-(img2[x2[j]+x,y2[j]+y]-mean2))**2
                    
            paireZMSDD[i,j]=ZMSDD
            

paire=[]



for i in range (len(x1)):
    if (np.min(paireZMSDD[i])<150000):
        paire1=i
        paire2=np.where(paireZMSDD[i]==np.min(paireZMSDD[i]))[0][0]
        paire.append((paire1,paire2))
        ax3.plot([y1[paire1],y2[paire2]+img1.shape[1]],[x1[paire1],x2[paire2]])
        
        
