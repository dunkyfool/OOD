#Save format
import numpy as np
import cv2
from tabulate import tabulate

filename = 'dog.jpg'
img_size = 480  # image weight/height
grid_num = 4    # grid cell number per side
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
  x=min(sx,ex)
  y=min(sy,ey)
  a = np.zeros((img_size,img_size))
  b = np.ones((h,w))
  c = []
  a[y:y+h,x:x+w] += b
#  print a
#  print b
  t=img_size/grid_num
  for i in range(grid_num):
    for j in range(grid_num):
      c += [min(1.0,a[i*t:i*t+t,j*t:j*t+t].sum())]
#    print c
#    print ' '
  return c

def transform(p_start,p_end,img_size,grid_num):
  #w,h = w/float(img_size),h/float(img_szie)
  bbox = []
  for i in range(len(p_start)):
    x=(float(p_start[i][0])+float(p_end[i][0]))/2
    y=(float(p_start[i][1])+float(p_end[i][1]))/2
    w=abs(float(p_start[i][0])-float(p_end[i][0])) + 1
    h=abs(float(p_start[i][1])-float(p_end[i][1])) + 1
    c=confidenceFinder(p_start[i],p_end[i],w,h,img_size,grid_num)

    x=x%float(img_size/grid_num)
    y=y%float(img_size/grid_num)
    w=w/float(img_size)
    h=h/float(img_size)

    x=x/float(img_size/grid_num)
    y=y/float(img_size/grid_num)
    bbox+=[(x,y,w,h,c)]
  return bbox

def draw_grid(img,img_size,grid_num):
  t=img_size/grid_num
  for i in range(grid_num):
    cv2.line(img,(t*(i+1),0),(t*(i+1),img_size-1),(0,0,0),1)
    cv2.line(img,(0,t*(i+1)),(img_size-1,t*(i+1)),(0,0,0),1)

def save2file(filename):
  f = open('img_label','a')
  output = transform(p_start,p_end,img_size,grid_num)
#  print len(output)
  for i in range(len(output)):
#    print output[i], type(output[i])
#    print output[i][0:4]
    f.write("%s " %filename)
    f.write("%s %s %s %s " %(output[i][0], output[i][1],
                            output[i][2], output[i][3]))
    for k in output[i][4]:
      f.write("%s " %k)
    f.write("\n")
  f.close()

########################
#        START         #
########################
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_bbox)

while(1):
    img = cv2.imread(filename)
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
    if cv2.waitKey(20) & 0xFF == 27:
        break
    elif cv2.waitKey(20) & 0xFF == ord('s'):
	#print tabulate(transform(p_start,p_end,img_size,grid_num),
        #       headers=['x','y','w','h','c'])
        print "Save to file..."
        save2file(filename)
    elif cv2.waitKey(20) & 0xFF == ord('m'):
        mode = not mode
        print mode
cv2.destroyAllWindows()

