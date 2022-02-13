# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 22:39:12 2022

@author: paulg
"""

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



def fast(I,N=9,t=80):

    h,l = I.shape
    t=80
    
    coins=[]
    V=np.zeros(I.shape)
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
                    
                if ((b>=N)|(d>=N)):
                    V[i,j]=sum(abs(FAST-Ic))
                    break
                
            
            
     
    FAST_max=np.zeros(I.shape)
                          
    largeur = 2
    for i in range(largeur,h-largeur):
        for j in range(largeur,l-largeur):
            valeur=V[i,j]
            if valeur==np.max(V[i-largeur:i+largeur+1,j-largeur:j+largeur+1]):
                FAST_max[i,j]=valeur
                V[i,j]=valeur+1
                   
    x,y=np.where(FAST_max>100)

    return (x,y,V)      


if __name__ == "__main__":
    img = Image.open("synthetic.gif")
    img = np.array(img)
    
    # img = Image.open("set1-1.png")
    # img = np.array(ImageOps.grayscale(img))
    
    
    # img = Image.open("set1-2.png")
    # img = np.array(ImageOps.grayscale(img))
    
    
    x,y,V = fast(img,N=9,t=80)
    
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("FAST")
    ax.plot(y,x,'+r')
    
    
    ax.imshow(img,cmap='gray')
                
            