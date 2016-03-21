import numpy as np
import cv2
import os
import sys
########################
#      VARIABLE        #
########################

record='changeName_record'         #save the next time index
start_index=0                      #default start index is zero
new_pathname='IR_train'            #rename folder
if ~os.path.isdir(new_pathname):   #if folder do not exist, create the new one
  cmd='mkdir '+new_pathname
  os.system(cmd)

########################
#     LOAD INDEX       #
########################

#Load Record to know where to start
#If the file is empty, then start from 1
#else start at the recorded number
if os.path.isfile(record):
  with open(record,'r') as f:
    start_index=int(f.read())
#print start_index, type(start_index)
#start_index+=1
title = 'Current index is '+str(start_index)+' Are you sure you want continue? (y/n):'
x = raw_input(title)
if x=='n':
  sys.exit()

########################
#     CHANGE NAME      #
########################

#Check program with folder name
if len(sys.argv)<=1:
  print "Retry the program like this: python changeName.py folder1 folder2 folder3 ..."
else:
  for pathname in sys.argv[1:]: #get folder name
#    print pathname
    files = [f for f in os.listdir(pathname)] # get file name under folder
#    print len(files)
#    print files
    for filename in files:
      _,_format = os.path.splitext(filename)
      filename = pathname+'/'+filename

      #fill zero to index
      if start_index/10<1:
        index = '00'+str(start_index)
      elif start_index/10>=1 and start_index/10<10:
        index = '0'+str(start_index)
      else:
        index = str(start_index)

      new_filename = new_pathname+'/'+index+_format
      cmd='cp '+filename+' '+new_filename
      os.system(cmd)
#      print filename, new_filename
#      img=cv2.imread(filename,0)
#      cv2.imshow('IMG',img)
#      cv2.waitKey(0)
#      cv2.destroyAllWindows()
      start_index+=1



########################
#     SAVE INDEX       #
########################
with open(record,'w') as f:
  f.write(str(start_index))

