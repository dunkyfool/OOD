import numpy as np


label = np.loadtxt('IR_train_label',dtype='str')
print label.shape

trainData = []
filenameList, trainLabels = label[:,0],label[:,1:].astype(np.int16)

print filenameList.shape
print trainLabels.shape

interval = 10 #10%
#interval = 5  #20%
#interval = 3  #30%
testNameList = filenameList[0::interval]
testLabels = trainLabels[0::interval]

filenameList = np.delete(filenameList,np.s_[::interval])
trainLabels = np.delete(trainLabels,np.s_[::interval],0)

print trainLabels.shape
print filenameList.shape

print testNameList.shape
print testLabels.shape

f = open('ir_train','a')
for i in range(trainLabels.shape[0]):
  f.write("%s "%filenameList[i])
  for j in range(trainLabels.shape[1]):
    f.write("%s "%trainLabels[i][j])
  f.write("\n")
f.close()

f = open('ir_test','a')
for i in range(testLabels.shape[0]):
  f.write("%s "%testNameList[i])
  for j in range(testLabels.shape[1]):
    f.write("%s "%testLabels[i][j])
  f.write("\n")
f.close()
