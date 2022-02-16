# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:07:02 2022

@author: paulg
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import *
import scipy.signal as sg
import cv2
import random

import match

img1 = Image.open("set1-1.png")
img1 = np.array(ImageOps.grayscale(img1))
img2 = Image.open("set1-2.png")
img2 = np.array(ImageOps.grayscale(img2))

x1,y1,x2,y2,couples = match.matching(img1,img2,'fast')


#%%
C=[]
for i in range(len(couples)):
    C.append([(x1[couples[i][0]],y1[couples[i][0]]),(x2[couples[i][1]],y2[couples[i][1]])])


Nmax=0
Hmax=[]
Cmax=[]
for j in range(100):

    c= random.sample(couples, k=4)
    
        
    A = [[x1[c[0][0]], y1[c[0][0]], 1, 0, 0, 0, -x2[c[0][1]]*x1[c[0][0]],-x2[c[0][1]]*y1[c[0][0]]],
         [0, 0, 0, x1[c[0][0]], y1[c[0][0]], 1, -y2[c[0][1]]*x1[c[0][0]], -y2[c[0][1]]*y1[c[0][0]]],
         [x1[c[1][0]], y1[c[1][0]], 1, 0, 0, 0, -x2[c[1][1]]*x1[c[1][0]],-x2[c[1][1]]*y1[c[1][0]]],
         [0, 0, 0, x1[c[1][0]], y1[c[1][0]], 1, -y2[c[1][1]]*x1[c[1][0]], -y2[c[1][1]]*y1[c[1][0]]],
         [x1[c[2][0]], y1[c[2][0]], 1, 0, 0, 0, -x2[c[2][1]]*x1[c[2][0]],-x2[c[2][1]]*y1[c[2][0]]],
         [0, 0, 0, x1[c[2][0]], y1[c[2][0]], 1, -y2[c[2][1]]*x1[c[2][0]], -y2[c[2][1]]*y1[c[2][0]]],
         [x1[c[3][0]], y1[c[3][0]], 1, 0, 0, 0, -x2[c[3][1]]*x1[c[3][0]],-x2[c[3][1]]*y1[c[3][0]]],
         [0, 0, 0, x1[c[3][0]], y1[c[3][0]], 1, -y2[c[3][1]]*x1[c[3][0]], -y2[c[3][1]]*y1[c[3][0]]]]
    
    A=np.array(A)
    
    b=np.array([x2[c[0][1]],y2[c[0][1]],x2[c[1][1]],y2[c[1][1]],x2[c[2][1]],y2[c[2][1]],x2[c[3][1]],y2[c[3][1]]])
    
    h=np.dot(np.linalg.inv(A),b) 
    
    H=[[h[0],h[1],h[2]],
       [h[3],h[4],h[5]],
       [h[6],h[7],1]]  
    
    
    
    n=0
    Cj=[]
    for i in range(len(C)):
        x=np.array([C[i][0][0],C[i][0][1],1])
        x_prime=np.array([C[i][1][0],C[i][1][1],1])
        
        Hx=np.dot(H,x)
        Hx=Hx/Hx[2]
        
        norme=np.linalg.norm(Hx-x_prime)
        
        if norme<3:
            n+=1
            Cj.append(C[i])
            
    if n>Nmax:
        Nmax=n
        Hmax=H
        Cmax=Cj

# A=np.zeros((2*Nmax,8))
# b=np.zeros((2*Nmax,1))
# for i in range(Nmax):
#     A[2*i,:]=[Cmax[i][0][0], Cmax[i][0][1], 1, 0, 0, 0, -Cmax[i][1][0]*Cmax[i][0][0],-Cmax[i][1][0]*Cmax[i][0][1]]
#     A[2*i+1,:]=[0, 0, 0, Cmax[i][0][0], Cmax[i][0][1], 1, -Cmax[i][1][1]*Cmax[i][0][0], -Cmax[i][1][1]*Cmax[i][0][1]]
#     b[2*i]=Cmax[i][1][0]
#     b[2*i+1]=Cmax[i][1][1]

# u,d,v=np.linalg.svd(A)

# b_prime=np.dot(np.transpose(u),b)
# y=b_prime/np.diag(d)

# H=np.dot(v,y)

h1,l1=img1.shape
h2,l2=img2.shape

img_full=np.concatenate((np.zeros(img1.shape),img2),axis=1)

for i in range(h1):
    for j in range(l1):
        x=np.array([i,j,1])
        Hx=np.dot(H,x)
        Hx=Hx/Hx[2]
        
        X,Y,Z=Hx
        if((X>=0)&(X<h1)):
            img_full[int(X),int(Y)+l1]=img1[i,j]
            
plt.imshow(img_full,cmap='gray')
        

        

