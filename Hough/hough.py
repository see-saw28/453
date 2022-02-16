# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 19:30:27 2022

@author: paulg
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import scipy.signal as sg
import cv2
import time

# img = Image.open("four.png")
# img = np.array(img)

img = Image.open("coins2.jpg")
img = np.array(ImageOps.grayscale(img))

#Filtrage de Sobel
start =time.time()
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

#Methode avec l'accumulateur
acc=np.zeros((600,600,900))

r,c = G.shape

for i in range(r):
    for j in range(c):
        if G_seuil[i,j]!=0:
            for x in range(r):
                for y in range(c):
                    if ((x!=i) & (y!=j)):
                        rad=int(np.sqrt((x-i)**2+(y-j)**2))+1
                        if (rad>=3):
                            acc[x,y,rad-3]+=139/rad #division par le rayon pour normaliser car le nombre de point est prop au rayon

#Maximum locaux
acc1=np.zeros((600,600,900)) 

larg=2 #on cherche le maximum dans un cube de largeur 2*larg+1                 
for i in range(larg,r-larg):
    for j in range(larg,c-larg):
        for k in range(larg,139-larg):
            valeur=acc[i,j,k]
            
            if valeur==np.max(acc[i-larg:i+larg+1,j-larg:j+larg+1,k-larg:k+larg+1]):
                acc1[i,j,k]=valeur 
                acc[i,j,k]=valeur+1 #evite de recopier plusieurs maximaux qui ont la même valeur

                
#Selection et affichage des N valeurs les plus grandes
N=4

img3 = Image.open("four3.png")
img3 = np.array(img3)

for i in range(N):
    r,c,rad=np.where(acc1==np.max(acc1))
    print(r,c,rad)
    cv2.circle(img3, center = (c[0]+1,r[0]+1), radius =rad[0]+3, color =(255,0,0), thickness=1)
    acc1[r,c,rad]=0
    plt.plot(c[0]+1,r[0]+1,'+r')
 
print(time.time()-start)
    
plt.imshow(img3)

