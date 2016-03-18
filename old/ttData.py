import numpy as np
import cv2
import os

#os.system('mkdir trainData')

img_size = 480
grid_size = 4
channel = 3
grid_l = img_size/grid_size
radius = 30
path = 'data/'

center = [grid_l/2 + i*grid_l for i in range(grid_size)]
#print center
for i in range(grid_size):
    for j in range(grid_size):
        img = np.zeros((img_size,img_size,channel), np.uint8)
        cv2.circle(img,(center[i],center[j]),radius,(255,255,255),-1)
#        cv2.imshow('image',img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        filename = path+str(i)+str(j)+'.jpg'
        cv2.imwrite(filename,img)
