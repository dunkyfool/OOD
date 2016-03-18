import numpy as np

filename='img_label'
label = np.loadtxt(filename,dtype='str')
#print label; print label.shape

#for i in range(label.shape[0]):
xywh = np.repeat(label[0][1:5].reshape((1,4)),16,axis=0)
#print xywh, xywh.shape
clas = np.repeat(label[0][21:].reshape((1,2)),16,axis=0)
#print clas, clas.shape

trainLabels = np.concatenate((xywh,label[0][5:21].reshape((16,1)),clas),axis=1)
print trainLabels, trainLabels.shape
