# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 09:40:50 2022

@author: paulg
"""
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
import itertools



def fast(I,N=9,t_min=80):
    
    t_max=120
    h,l = I.shape
    
    
    coins=[]
    T=np.zeros(I.shape)
    for i in range(3,h-3):
        for j in range(3,l-3):
            FAST=[I[i+3,j],I[i+3,j+1],I[i+2,j+2],I[i+1,j+3],I[i,j+3],I[i-1,j+3],I[i-2,j+2],I[i-3,j+1],I[i-3,j],I[i-3,j-1],I[i-2,j-2],I[i-1,j-3],I[i,j-3],I[i+1,j-3],I[i+2,j-2],I[i+3,j-1]]
            Ic=I[i,j]
            FAST1=FAST-Ic
            FAST2=np.concatenate((FAST1,FAST1)).astype('int32')
            n=16
            t=t_min
            # print(FAST1)
            while n>=N:
                condition=(FAST2-t)>0
                cond=[ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
                # print(FAST2-t)
                if (cond!=[]):
                    n1=max(cond)
                else :
                    n1=0
                condition=(FAST2+t)<0
                cond=[ sum( 1 for _ in group ) for key, group in itertools.groupby( condition ) if key ]
                if (cond!=[]):
                    n2=max(cond)     
                else:
                    n2=0
                n=max(n1,n2)
                
                t+=1
                if t==t_max:
                    break
            if t==t_min+1:
                T[i,j] =0
            else:
                T[i,j]=t
                
            print(n,t)
            
            
            
     
    FAST_max=np.zeros(I.shape)
                          
    largeur = 2
    for i in range(largeur,h-largeur):
        for j in range(largeur,l-largeur):
            valeur=T[i,j]
            if valeur==np.max(T[i-largeur:i+largeur+1,j-largeur:j+largeur+1]):
                FAST_max[i,j]=valeur
                T[i,j]=valeur+1
                   
    x,y=np.where(FAST_max>80)

    return (x,y,T)      


if __name__ == "__main__":
    img = Image.open("synthetic.gif")
    img = np.array(img)
    
    # img = Image.open("set1-1.png")
    # img = np.array(ImageOps.grayscale(img))
    
    
    # img = Image.open("set1-2.png")
    # img = np.array(ImageOps.grayscale(img))
    
    
    x,y,V = fast(img,N=9)
    
    fig, ax = plt.subplots(1, 1)
    fig.suptitle("FAST")
    ax.plot(y,x,'+r')
    
    
    ax.imshow(img,cmap='gray')
                
            