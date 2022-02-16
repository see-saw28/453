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

x1,y1,x2,y2,couples = match(img1,img2,'fast')

couples_choisis= sample(couples, k=4)