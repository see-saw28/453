# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:58:01 2022

@author: paulg
"""

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



def fast(I,t=80):

    h,l = I.shape
    t=80
    
    coins=[]
    maxi=np.zeros(I.shape)
    
    for i in range(3,h-3):
        for j in range(3,l-3):
            FAST=[I[i+3,j],I[i+3,j+1],I[i+2,j+2],I[i+1,j+3],I[i,j+3],I[i-1,j+3],I[i-2,j+2],I[i-3,j+1],I[i-3,j],I[i-3,j-1],I[i-2,j-2],I[i-1,j-3],I[i,j-3],I[i+1,j-3],I[i+2,j-2],I[i+3,j-1]]
            d=0
            b=0
            dmax=0
            bmax=0
            Ic=I[i,j]
            fast=np.zeros((16,1))
            last=''
            FAST2=np.concatenate((FAST,FAST))
            for k in range(len(FAST2)):
                
                if (FAST2[k]<Ic-t):
                    d+=1
                    if (last!='d'):
                        if (b>bmax):
                            bmax=b
                        b=0
                    last='d'
                        
                elif (FAST2[k]>Ic+t):
                    b+=1
                    if (last!='b'):
                        if (d>dmax):
                            dmax=d
                        d=0
                    last='b'
                     
                else :
                    last='s'
                    if (d>dmax):
                        dmax=d
                    if (b>bmax):
                        bmax=b
                    d=0
                    b=0
                    
            maxi[i,j]=max(dmax,bmax)
     
    FAST_max=np.zeros(I.shape)
                          
    largeur = 2
    for i in range(largeur,h-largeur):
        for j in range(largeur,l-largeur):
            valeur=maxi[i,j]
            if valeur==np.max(maxi[i-largeur:i+largeur+1,j-largeur:j+largeur+1]):
                FAST_max[i,j]=valeur
                maxi[i,j]=valeur+1
                   
    x,y=np.where(FAST_max>9)

    return (x,y)      


if __name__ == "__main__":
    # img = Image.open("synthetic.gif")
    # iimg = np.array(img)
    
    # img = Image.open("set1-1.png")
    # img = np.array(ImageOps.grayscale(img))
    
    
    img = Image.open("set1-2.png")
    img = np.array(ImageOps.grayscale(img))
    
    
    x,y = fast(img,t=80)
    
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("FAST")
    ax.plot(y,x,'+r')
    
    
    ax.imshow(img,cmap='gray')
                
            