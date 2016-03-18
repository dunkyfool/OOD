import numpy as np
import cv2
import os

def load_image(folder):
  images = []
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
      images+=[img]
  return images

print os.listdir("/home/hduser/OOD/data/")
