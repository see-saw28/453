# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:05:26 2022

@author: paulg
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import scipy.signal as sg
import cv2



img = Image.open("synthetic.gif")
I = np.array(img)

h,l = I.shape
t=20

coins=[]

for i in range(3,h-3):
    for j in range(3,l-3):
        FAST=[I[i+3,j],I[i+3,j+1],I[i+2,j+2],I[i+1,j+3],I[i,j+3],I[i-1,j+3],I[i-2,j+2],I[i-3,j+1],I[i-3,j],I[i-3,j-1],I[i-2,j-2],I[i-1,j-3],I[i,j-3],I[i+1,j-3],I[i+2,j-2],I[i+3,j-1]]
        d=0
        b=0
        Ic=I[i,j]
        fast=np.zeros((16,1))
        last=''
        FAST2=np.concatenate((FAST,FAST))
        for k in range(len(FAST2)):
            
            if (FAST2[k]<Ic-t):
                d+=1
                if (last!='d'):
                    b=0
                last='d'
                    
            elif (FAST2[k]>Ic+t):
                b+=1
                if (last!='b'):
                    d=0
                last='b'
                 
            else :
                last='s'
                d=0
                b=0
                
            if ((b>8)|(d>8)):
                coins.append([i,j])
                break
                
coins=np.array(coins)

x=coins[:,0]
y=coins[:,1]         

fig, ax = plt.subplots(1, 1)
fig.suptitle("FAST")
ax.plot(y,x,'+r')


ax.imshow(img,cmap='gray')
                
            