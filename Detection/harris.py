# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:53:08 2022

@author: paulg
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import *
import scipy.signal as sg
import cv2




def harris(img,seuil=0.6,taille_fenetre=4,sigma = 3):
    #Filtrage de Sobel

    Hx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Hy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

    Gx = sg.convolve2d(img,Hx,'valid')
    Gy = sg.convolve2d(img,Hy,'valid')
    
    G=np.sqrt(np.square(Gx)+np.square(Gy))
    
    
    
    h,l=Gx.shape
    img_R=np.zeros(Gx.shape)
    
    
    
    alpha=0.05
    
    def w(x,y):
        return np.exp(-(x**2+y**2)/(2*sigma**2))
    
    for i in range(h):
        for j in range(l):
            M=np.zeros((2,2))
            for x in range(-taille_fenetre,taille_fenetre+1):
                for y in range(-taille_fenetre,taille_fenetre+1):
                    if (((i+x)>=0)&((i+x)<h)&((j+y)>=0)&((j+y)<l)):
                        M+=w(x,y)*np.array([[Gx[i+x,j+y]**2,Gx[i+x,j+y]*Gy[i+x,j+y]],[Gx[i+x,j+y]*Gy[i+x,j+y],Gy[i+x,j+y]**2]])
                        
            
            R=np.linalg.det(M)-alpha*np.trace(M)**2
            img_R[i,j]=R
            
            
    
    R_raw=img_R/np.max(img_R)
           
    
    
    R=(R_raw>seuil)*R_raw 
    
    R_max=np.zeros(R.shape)
                        
    largeur = 2
    for i in range(largeur,h-largeur):
        for j in range(largeur,l-largeur):
            valeur=R[i,j]
            if valeur==np.max(R[i-largeur:i+largeur+1,j-largeur:j+largeur+1]):
                R_max[i,j]=valeur
    
    x,y=np.where(R_max>seuil)
    
    return (R_raw,x,y)

if __name__ == "__main__":
    
    img = Image.open("synthetic.gif")
    img = np.array(img)

    seuil=0.6
    R_max,x,y=harris(img,seuil)
    plt.plot(y,x,'+r')
    plt.imshow(img,cmap='gray')
         
        