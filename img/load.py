import numpy as np
import cv2
import os

#l=[]
#l+=[os.system('ls photo')]
#print len(l)
#print l


l = [f for f in os.listdir('photo/')]
print len(l)
print l

#img = cv2.imread('frame54.jpg',0)
#print img.shape
#cv2.imshow('IMG',img)
#cv2.waitKey(0)& 0xFF
#cv2.destroyAllWindows()
