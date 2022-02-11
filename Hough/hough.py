# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:30:27 2022

@author: paulg
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import *
import scipy.signal as sg
import cv2


img = Image.open("fourn.png")
img = np.array(img)

#Filtrage de Sobel

Hx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Hy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])


Gx = sg.convolve2d(img,Hx,'valid')
Gy = sg.convolve2d(img,Hy,'valid')

G=np.sqrt(np.square(Gx)+np.square(Gy))

# plt.imshow(G,cmap='gray')
# plt.show()


seuil = 500
G_seuil = (G>seuil)*G

# plt.imshow(G_seuil,cmap='gray')
# plt.show()


acc=np.zeros((100,100,141))

r,c = G.shape

for i in range(r):
    for j in range(c):
        if G_seuil[i,j]!=0:
            for x in range(r):
                for y in range(c):
                    if ((x!=i) & (y!=j)):
                        rad=int(np.sqrt((x-i)**2+(y-j)**2))
                        acc[x,y,rad]+=141/rad

acc1=np.zeros((100,100,141))
                    
for i in range(1,r-1):
    for j in range(1,c-1):
        for k in range(1,141-1):
            valeur=acc[i,j,k]
            if valeur==np.max(acc[i-1:i+2,j-1:j+2,k-1:k+2]):
                acc1[i,j,k]=valeur
N=4
img3 = Image.open("fourn3.png")
img3 = np.array(img3)

for i in range(N):
    r,c,rad=np.where(acc==np.max(acc1))
    print(r,c,rad)
    cv2.circle(img3, center = (c[0],r[0]), radius =rad[0], color =(255,0,0), thickness=1)
    acc1[r,c,rad]=0
    
    
    
plt.imshow(img3)