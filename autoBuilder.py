import numpy as np
from itertools import izip
import cv2
import os

def confidenceFinder(cx,cy,r,img_size,grid_num):
  #array(row,column) vs img(x,y)
  grid_pixels = (img_size/grid_num)**2
  x = cx-r
  y = cy-r
  a = np.zeros((img_size,img_size))
  b = np.ones((2*r,2*r))

#  A = grid_pixels #Grid total pixels
#  B = b.sum() #BBox total pixels
  c = []
  a[y:y+2*r,x:x+2*r] += b
#  print a
#  print b
  t=img_size/grid_num
  for i in range(grid_num):
    for j in range(grid_num):
      if a[i*t:(i+1)*t,j*t:(j+1)*t].sum() > grid_pixels * 0.1:
        c += [1]
      else:
        c += [0]
  C = np.asarray(c).reshape((grid_num,grid_num))
#  print x,y,cx,cy
  print C
  print c
#  print ' '
  return c

def draw_grid(img,img_size,grid_num):
  t=img_size/grid_num
  for i in range(grid_num):
    cv2.line(img,(t*(i+1),0),(t*(i+1),img_size-1),(255,255,255),1)
    cv2.line(img,(0,t*(i+1)),(img_size-1,t*(i+1)),(255,255,255),1)

def save2file(filename,img_name,output):
  f = open(filename,'a')
#  print len(output)
#    print output[i], type(output[i])
#    print output[i][0:4]
  f.write("%s " %img_name)
  for i in range(len(output)):
    f.write("%s " %(output[i]))
  f.write("\n")

# Create images
def createImage():
  filename='oracleTest'
  img_size=160
  grid_size=10
  channel=3
  radius=16
  path='data/oracleTest/'
#  center=[radius+i*radius/2 for i in range((img_size/radius-1)*2-1)]
  x = np.random.randint(radius+1,img_size-radius-1,10)
  y = np.random.randint(radius+1,img_size-radius-1,10)
#  for i in center:
#    for j in center:
  for i in x:
    for j in y:
      #print i,j
# Find object confidence score
      output = confidenceFinder(i,j,radius,img_size,grid_size)
      img = np.zeros((img_size,img_size,channel), np.uint8)
      cv2.circle(img,(i,j),radius,(255,255,255),-1)
#      draw_grid(img,img_size,grid_size)
#      cv2.imshow('image',img)
#      cv2.waitKey(0)
#      cv2.destroyAllWindows()
      img_name = path+str(i)+str(j)+'.jpg'
      cv2.imwrite(img_name,img)
      save2file(filename,img_name,output)


# Save Image & Labels


if __name__=='__main__':
  createImage()
