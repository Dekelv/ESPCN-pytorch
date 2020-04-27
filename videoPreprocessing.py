from PIL import Image
import cv2
from os import listdir
import sys, h5py
import glob
from multiprocessing import Pool
import numpy as np

def SaveFrames(VideoFile, outputDir): 
    print(cv2.__version__)
    vidcap = cv2.VideoCapture(VideoFile)
    success,image = vidcap.read()
    count = 0
    cnt = "0000"
    while success:
        cnt= cnt[:-len(str(count))]+str(count)
        cv2.imwrite(outputDir+"/frame"+cnt+".jpg", image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1


        
