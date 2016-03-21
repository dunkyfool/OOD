#Save format
import numpy as np
import cv2
import os
from tabulate import tabulate

#filename = 'photo/frame54.jpg'
pathname = 'IR_train'
savefile = 'IR_train_label'
img_size = 320  # image weight/height
grid_num = 10   # grid cell number per side
mx,my = 0,0	# drawing mouse location
p_start=[]	# start point
p_end=[]	# end point
drawing = False
mode = False

def draw_bbox(event,x,y,flags,param):
    global ix,iy,mx,my,p_start,p_end,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        if x<0: x=0
        if y<0: y=0
        if x>img_size: x=img_size-1
        if y>img_size: y=img_size-1
        p_start+=[(x,y)]
        print len(p_start), p_start
        ix,iy,mx,my = x,y,x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
          mx,my = x,y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if x<0: x=0
        if y<0: y=0
        if x>img_size: x=img_size-1
        if y>img_size: y=img_size-1
        p_end+=[(x,y)]
        print len(p_end),p_end
    elif event == cv2.EVENT_RBUTTONUP:
        if len(p_start) !=0:
            print len(p_start),p_start.pop()
            print len(p_end),p_end.pop()

def confidenceFinder((sx,sy),(ex,ey),w,h,img_size,grid_num):
  #array(row,column) vs img(x,y)
  grid_pixels = (img_size/grid_num)**2
  x=min(sx,ex)
  y=min(sy,ey)
  xx=max(sx,ex)
  yy=max(sy,ey)
  a = np.zeros((img_size,img_size))
  b = np.ones((h,w))

  A = grid_pixels #Grid total pixels
  B = b.sum() #BBox total pixels
  c = []
  a[y:y+h,x:x+w] += b
#  print a
#  print b
  t=img_size/grid_num
  for i in range(grid_num):
    for j in range(grid_num):
#      x_in = True if x+xx >= 2*(i*t) and x+xx <= 2*(i*t+t) else False
#      y_in = True if y+yy >= 2*(j*t) and y+yy <= 2*(j*t+t) else False
#      A_B = a[i*t:i*t+t,j*t:j*t+t].sum()
#      if A==B and x_in and y_in: #if BBox within the grid cell
#        IOU = A_B/ float(A)
#        c += [IOU]
#      else:
#        IOU = A_B/ float(A + B - A_B)
#        c += [IOU]
      if a[i*t:i*t+t,j*t:j*t+t].sum() > grid_pixels * 0.1:
        #c += [min(1.0,a[i*t:i*t+t,j*t:j*t+t].sum())]
        c += [1]
      else:
        c += [0]
  print c
  print ' '
  return c

def transform(p_start,p_end,img_size,grid_num):
#  #w,h = w/float(img_size),h/float(img_szie)
  bbox = []
  if len(p_start)==0:
    c=np.zeros(100,dtype=np.uint8).tolist()
    bbox+=[c]
  for i in range(len(p_start)):
#    x=(float(p_start[i][0])+float(p_end[i][0]))/2
#    y=(float(p_start[i][1])+float(p_end[i][1]))/2
    w=abs(float(p_start[i][0])-float(p_end[i][0])) + 1
    h=abs(float(p_start[i][1])-float(p_end[i][1])) + 1
    c=confidenceFinder(p_start[i],p_end[i],w,h,img_size,grid_num)
#
#    x=x%float(img_size/grid_num)
#    y=y%float(img_size/grid_num)
#    w=w/float(img_size)
#    h=h/float(img_size)
#
#    x=x/float(img_size/grid_num)
#    y=y/float(img_size/grid_num)
#    bbox+=[(x,y,w,h,c)]
    bbox+=[c]
  return bbox

def draw_grid(img,img_size,grid_num):
  t=img_size/grid_num
  for i in range(grid_num):
    cv2.line(img,(t*(i+1),0),(t*(i+1),img_size-1),(255,255,255),1)
    cv2.line(img,(0,t*(i+1)),(img_size-1,t*(i+1)),(255,255,255),1)

def save2file(filename,savefile):
  f = open(savefile,'a')
  output = transform(p_start,p_end,img_size,grid_num)
#  print len(output), type(output);raw_input()
  for i in range(len(output)):
#    print output[i], type(output[i])
#    print output[i][0:4]
    f.write("%s " %filename)
#    f.write("%s %s %s %s " %(output[i][0], output[i][1],
#                            output[i][2], output[i][3]))
    for k in output[i]:
      f.write("%s " %k)
    f.write("\n")
  f.close()

########################
#        START         #
########################

#load files in folder

files = [f for f in os.listdir(pathname)]
#print len(files)
#print files
for filename in files:
  filename = os.path.join(pathname,filename)
#  img = cv2.imread(os.path.join(pathname,filename))
#  print filename, img.shape

#playout
  cv2.namedWindow('image')
  cv2.setMouseCallback('image',draw_bbox)

  while(1):
      img = cv2.imread(filename)
#      print img;raw_input()
      img = cv2.resize(img, (img_size,img_size))
      # Draw bbox
      if len(p_end) !=0 and not drawing:
          for i in range(len(p_start)):
              cv2.rectangle(img,p_start[i],p_end[i],(0,255,0),3)
      if drawing :
          cv2.rectangle(img,(ix,iy),(mx,my),(255,0,0),3)
      if mode:
          draw_grid(img,img_size,grid_num)

      cv2.imshow('image',img)

      # Command
      key = cv2.waitKey(10) & 0xFF
      if key == ord('s'):
            #print tabulate(transform(p_start,p_end,img_size,grid_num),
            #       headers=['x','y','w','h','c'])
          print "Save to file..."
          save2file(filename,savefile)
      elif key == 27:
          break
      elif key == ord('m'):
          mode = not mode
          print mode
  cv2.destroyAllWindows()
  del p_start[:]
  del p_end[:]

