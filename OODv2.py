#--------------------------------------------------------------------------
#
#  Project Nmae: OODv2.py
#  Goal: Build a single neural network to do object detection
#
#
#  Date: 2016.03.16
#  Author: Jackie
#--------------------------------------------------------------------------
#  Warning:
#  Cost function (slow for loop)
#  verify accurarcy
#  image size in this code and dataBuilder must be the same
#  test123
import theano, theano.tensor as T
import scipy.io as sio
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from itertools import izip
import timeit, cPickle, pylab
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
import cv2
import os

##########################
#   Actvation Function   #
##########################
def ReLU(x):
  y = T.maximum(0.001*x,x)
  return y

def Sigmoid(x):
  y = T.nnet.sigmoid(x)
  return y

def tanh(x):
  y = T.tanh(x)
  return y

def Softmax(x):
  y = T.nnet.softmax(x)
  return y

##########################
#     Cost Function      #
##########################
def ED(x,y):
  z = T.sum((x-y)**2)
  return z

def NLL(y,y_hat):
  row = y.shape[0]
  k = y[T.arange(row), y_hat.argmax(axis=1)]
  return -T.mean(T.log(k))

def YOLO(y,y_hat,batch_size,grid_size,class_num):
  #slow for loop
  grid_sq = grid_size**2
  lambda_score = 10.0
  lambda_xywh = 3.0
  lambda_class = 0.0001
  y = y.reshape((batch_size,grid_sq,5+class_num))
  y_hat = y_hat.reshape((batch_size,grid_sq,5+class_num))

  score = (y[:,:,4]-y_hat[:,:,4])**2
  cost = lambda_score * T.sum(score)
  for i in range(4):
    tmp = (y[:,:,i]-y_hat[:,:,i])**2
    cost += lambda_xywh * T.sum(tmp)

  for i in range(class_num):
    tmp = (y[:,:,5+i]-y_hat[:,:,5+i])**2
    cost += lambda_class * T.sum(tmp)
  return cost

def OBJ(y,y_hat):
  z=T.sum(T.switch(T.eq(y_hat,1),5*(y-y_hat)**2,0.5*(y-y_hat)**2))
  return z


##########################
#    Record Function     #
##########################
def record(filename,option,params):
  f = open(filename,'a')
  if option == 1:
    f.write("%.2f %d %d %d %.7f %d\n" %(params[1],params[0],params[2],
                                        params[3],params[4],params[5]))
  elif option == 2: #Add L1,L2
    f.write("%.2f %.2f %d %d %d %.7f %d %.10f %.10f\n" %(params[1],params[8],
     params[0],params[2],params[3],params[4],params[5],params[6],params[7]))
  elif option == 3: #log YOLO
    f.write(params)
    f.write("\n")
  elif option == 4:
    f.close()
    f = open(filename,'w')
    f.write(params)
    f.write("\n")
  f.close()

##########################
# Save params Function   #
##########################
def save_params(filename, params=[]):
  save_file = open(filename,'wb')
  for i in params:
    cPickle.dump(i,save_file,-1)
  save_file.close()

##########################
#    Shuffle Function    #
##########################
def shuffle(a,b,index):
  z=np.append(a,b,axis=1)
  z=np.random.permutation(z)
  return z[:,0:index], z[:,index:]

##########################
#    Update Function     #
##########################
def MyUpdate(parameters,gradients,mu,wd):
  parameters_updates = []
  for p,g in izip(parameters,gradients):
    parameters_updates = parameters_updates + [(p,p*(1-wd*mu)-mu*g)]
  return parameters_updates

##########################
#       DNN_Layer        #
##########################
class HiddenLayer(object):
  def __init__(self, input, n_in, n_out, n_batch, act):
    self.input = input
    self.n_batch = n_batch
    self.act = act
    w_values = np.asarray(np.random.uniform(
                -4*np.sqrt(6. / (n_in+n_out)),
                 4*np.sqrt(6. / (n_in+n_out)),
                 size=(n_in,n_out)))
    w = theano.shared(value=w_values,name='w')
    b_values = np.asarray(np.zeros((n_out,)))
    b = theano.shared(value=b_values,name='b')
    self.w = w
    self.b = b
    lin_output = T.dot(input,self.w)+self.b
    self.output = act(lin_output)
    self.params = [self.w, self.b]

##########################
#       CNN_Layer        #
##########################
class CNN_Layer(object):
  def __init__(self, input, filter_shape, image_shape, poolsize=(2,2), poolFlag=True, border_mode=True):
    self.input = input
    fan_in = np.prod(filter_shape[1:])
    fan_out = np.prod(filter_shape[0]*np.prod(filter_shape[2:]/np.prod(poolsize)))

    w_bound = np.sqrt(6./(fan_in+fan_out))
    w_values = np.asarray(np.random.uniform(-w_bound,w_bound,size=filter_shape))
    w = theano.shared(value=w_values,name='W')
    b_values = np.asarray(np.zeros((filter_shape[0],)))
    b = theano.shared(value=b_values,name='B')
    self.w = w
    self.b = b
    if border_mode:
      conv_out = conv.conv2d(input,self.w,border_mode='full')
    else:
      conv_out = conv.conv2d(input,self.w,border_mode='valid')
#    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    output = ReLU(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    pool_out = downsample.max_pool_2d(output, poolsize, ignore_border=True)
    if poolFlag:
      self.output = pool_out
    else:
      self.output = output
    self.params = [self.w, self.b]

##########################
#          MLP           #
##########################
class MLP(object):
  def __init__(self,input,y_hat,n_in,n_hidden,n_out,n_batch):
    self.L1 = HiddenLayer(input,n_in,n_hidden,n_batch,Sigmoid)
    self.L2 = HiddenLayer(self.L1.output,n_hidden,n_out,n_batch,Sigmoid)
    self.params = self.L1.params + self.L2.params
    self.output = self.L2.output

##########################
#        CNN-MLP         #
##########################
class CNN_MLP(object):
  def __init__(self, input, filter_size, img_size, kernel, batch_size,poolFlag,border_mode):
    self.input = input.reshape((batch_size,kernel[0],img_size,img_size))
    self.L1 = CNN_Layer(self.input,
                  filter_shape=(kernel[1],kernel[0],filter_size[0],filter_size[0]),
                  image_shape=(batch_size,kernel[0],img_size,img_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[0],
                  border_mode=border_mode[0])
    if border_mode[0]:
      tmp_size = img_size + filter_size[0] - 1
    else:
      tmp_size = img_size - filter_size[0] + 1
    if poolFlag[0]:
      tmp_size /= 2

    self.L2 = CNN_Layer(self.L1.output,
                  filter_shape=(kernel[2],kernel[1],filter_size[1],filter_size[1]),
                  image_shape=(batch_size,kernel[1],tmp_size,tmp_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[1],
                  border_mode=border_mode[1])
    if border_mode[1]:
      tmp_size = tmp_size + filter_size[1] - 1
    else:
      tmp_size = tmp_size - filter_size[1] + 1
    if poolFlag[1]:
      tmp_size /= 2

    self.L3 = CNN_Layer(self.L2.output,
                  filter_shape=(kernel[3],kernel[2],filter_size[2],filter_size[2]),
                  image_shape=(batch_size,kernel[2],tmp_size,tmp_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[2],
                  border_mode=border_mode[2])
    if border_mode[2]:
      tmp_size = tmp_size + filter_size[2] - 1
    else:
      tmp_size = tmp_size - filter_size[2] + 1
    if poolFlag[2]:
      tmp_size /= 2

    self.L4 = CNN_Layer(self.L3.output,
                  filter_shape=(kernel[4],kernel[3],filter_size[3],filter_size[3]),
                  image_shape=(batch_size,kernel[3],tmp_size,tmp_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[3],
                  border_mode=border_mode[3])
    if border_mode[3]:
      tmp_size = tmp_size + filter_size[3] - 1
    else:
      tmp_size = tmp_size - filter_size[3] + 1
    if poolFlag[3]:
      tmp_size /= 2

#    self.L5 = CNN_Layer(self.L4.output,
#                  filter_shape=(kernel[5],kernel[4],filter_size[4],filter_size[4]),
#                  image_shape=(batch_size,kernel[4],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[4],
#                  border_mode=border_mode[4])
#    if border_mode[4]:
#      tmp_size = tmp_size + filter_size[4] - 1
#    else:
#      tmp_size = tmp_size - filter_size[4] + 1
#    if poolFlag[4]:
#      tmp_size /= 2
#
#    self.L6 = CNN_Layer(self.L5.output,
#                  filter_shape=(kernel[6],kernel[5],filter_size[5],filter_size[5]),
#                  image_shape=(batch_size,kernel[5],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[5],
#                  border_mode=border_mode[5])
#    if border_mode[5]:
#      tmp_size = tmp_size + filter_size[5] - 1
#    else:
#      tmp_size = tmp_size - filter_size[5] + 1
#    if poolFlag[5]:
#      tmp_size /= 2
#
#    self.L7 = CNN_Layer(self.L6.output,
#                  filter_shape=(kernel[7],kernel[6],filter_size[6],filter_size[6]),
#                  image_shape=(batch_size,kernel[6],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[6],
#                  border_mode=border_mode[6])
#    if border_mode[6]:
#      tmp_size = tmp_size + filter_size[6] - 1
#    else:
#      tmp_size = tmp_size - filter_size[6] + 1
#    if poolFlag[6]:
#      tmp_size /= 2
#
#    self.L8 = CNN_Layer(self.L7.output,
#                  filter_shape=(kernel[8],kernel[7],filter_size[7],filter_size[7]),
#                  image_shape=(batch_size,kernel[7],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[7],
#                  border_mode=border_mode[7])
#    if border_mode[7]:
#      tmp_size = tmp_size + filter_size[7] - 1
#    else:
#      tmp_size = tmp_size - filter_size[7] + 1
#    if poolFlag[7]:
#      tmp_size /= 2

#    self.L4 = CNN_Layer(self.L3.output,
#                  filter_shape=(kernel[4],kernel[3],filter_size[1],filter_size[1]),
#                  image_shape=(batch_size,kernel[3],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[3],
#                  border_mode=border_mode[3])
#    if border_mode[3]:
#      tmp_size = tmp_size + filter_size[3] - 1
#    else:
#      tmp_size = tmp_size - filter_size[3] + 1
#    if poolFlag[3]:
#      tmp_size /= 2

    self.output_size = kernel[4]*tmp_size**2
#    print self.output_size,kernel[6],tmp_size
#    raw_input('STOP')
    self.params = self.L1.params + self.L2.params + self.L3.params + self.L4.params# + self.L5.params + self.L6.params #+ self.L7.params + self.L8.params
    self.output = self.L4.output.reshape((batch_size,self.output_size))

def printOutput(last_params,trainData,testData,layer_num,filter_size,img_size,batch_size,kernel,poolFlag,border_mode,label):
  X = T.matrix('X')
  tmp_size = img_size
  total = trainData.shape[0] + testData.shape[0]
  output=[]
  for i in range(layer_num):
    if i!=0:
      if poolFlag[i-1]:
        tmp_size = (tmp_size - filter_size[i-1] + 1)/2
      else:
        tmp_size = (tmp_size - filter_size[i-1] + 1)
#    print batch_size,kernel[i],tmp_size
#    raw_input()
    X = X.reshape((batch_size,kernel[i],tmp_size,tmp_size))
    tmp = CNN_Layer(X,
                    filter_shape=(kernel[i+1],kernel[i],filter_size[i],filter_size[i]),
                    image_shape=(batch_size,kernel[i],tmp_size,tmp_size),
                    poolsize=(2,2),
                    poolFlag=poolFlag[i],
                    border_mode=border_mode[i])
    f=theano.function(inputs=[X],outputs=[tmp.output])
    tmp.w.set_value(last_params[(layer_num-i)*(-2)])
    tmp.b.set_value(last_params[(layer_num-i)*(-2)+1])
    if i==0:
      for j in range(trainData.shape[0]):
        output+=[ f(trainData[j:j+1].reshape((1,1,tmp_size,tmp_size)))[0] ]
      for j in range(testData.shape[0]):
        output+=[ f(testData[j:j+1].reshape((1,1,tmp_size,tmp_size)))[0] ]
#      for k in range(len(output)):
#        print k
#        print output[k]
#        print output[k].shape
        #print output[k].argmax()
#        raw_input()
    else:
      for j in range(total):
        if i==layer_num-1:
          print j
          print f(output[-total])[0]
          print f(output[-total])[0].shape
          print 'Predict: '+str(f(output[-total])[0].argmax())
          print 'Answer:  '+str(label[j].argmax())
          raw_input()
        output+=[ f(output[-total])[0] ]

##########################
#      Load Data         #
##########################
def loadData(filename,grid_num,img_size):
  #trainLabels(training_num, grid_sq, xywhC)
  #trainData will follow img_label to build the trainData
  label = np.loadtxt(filename,dtype='str')
  grid_sq = grid_num **2
  trainData = []
  #print label; print label.shape
  #raw_input()
  filenameList, trainLabels = label[:,0],label[:,1:].astype(np.int16)
#  print filenameList,filenameList.shape
#  print trainLabels,trainLabels.shape
#  raw_input()
  for i in range(filenameList.shape[0]):
    #img = cv2.imread(os.path.join('data/',label[i][0]))
    img = cv2.imread(filenameList[i])
    img = cv2.resize(img,(img_size,img_size))
    img = img.astype(np.float64)/255.
#    print img.shape
#    raw_input()
#    filenameList+=[label[i][0]]
    #cv2.imshow('img',img)
    #cv2.waitKey(0) & 0xFF
    #cv2.destroyAllWindows()
    trainData += [img.reshape((3*img_size**2))]
  trainData = np.asarray(trainData)
#  print trainData.shape
#  raw_input()

#  print trainData[0]
  trainData -= trainData.mean()
#  print trainData[0]
#  raw_input()
  return filenameList,trainData, trainLabels

##########################
# Train & Valid function #
##########################
def trainNetwork(g,v,trainData,trainLabels,batch_size,epoch_num,img_size,grid_size,dnn,cnn,testData,testLabels,filenameList,test_filenameList,filename):
  good_TrainScore = 0
  good_TestScore = 0
  start_time = timeit.default_timer()
  for e in range(epoch_num):
    for i in range(trainData.shape[0]/batch_size):
      y,c = g(trainData[i*batch_size:(i+1)*batch_size],
              trainLabels[i*batch_size:(i+1)*batch_size])
    print("Epoch=%3d & cost= %.3f" %(e+1,c))
    #shuffle(trainData,trainLabels,channel*img_size**2) #import img_size and channel
##########################
#       Validation       #
##########################
    if (e+1)%10==0:
      trainScoreCtr = 0
      testScoreCtr = 0
      print "Training Set Wrong Image:"
      for x in range(trainLabels.shape[0]):
        output = v(trainData[x:x+1])
        predict = (output[0]>0.5)
        answer = (trainLabels[x:x+1]==1)
#        print trainLabels[x:x+1].shape
#        print predict.shape
#        print answer.shape
#        raw_input()
        #print predict.sum(),answer.sum()
        if (predict==answer).all():
          trainScoreCtr += 1
        else:
          print filenameList[x]
      print "Testing Set Wrong Image:"
      for x in range(testLabels.shape[0]):
        output = v(testData[x:x+1])
        predict = (output[0]>0.5)
        answer = (testLabels[x:x+1]==1)
        if (predict==answer).all():
          testScoreCtr += 1
        else:
          print test_filenameList[x]
      currentTrainScore = trainScoreCtr*100./trainLabels.shape[0]
      currentTestScore = testScoreCtr*100./testLabels.shape[0]
      print("Train Accurarcy: %.3f%%; Test Accurarcy: %.3f%%" %(currentTrainScore,currentTestScore))
#      print trainScoreCtr, good_scoreCtr, trainAccDelta, good_accDelta
#      print testScoreCtr, good_scoreCtr, testAccDelta, good_accDelta
      if currentTrainScore >= good_TrainScore and currentTestScore >= good_TestScore:
        print "SAVE PARAMETERS!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        good_TrainScore = currentTrainScore
        good_TestScore = currentTestScore
        good_record = 'Train: '+str(good_TrainScore)+',Test: '+str(good_TestScore)
        log = filename+'_good_record'
        name = filename+'_para'
        record(log,4,good_record)
        save_params(name, params=[ dnn.L1.w.get_value(),
                                   dnn.L1.b.get_value(),
                                   dnn.L2.w.get_value(),
                                   dnn.L2.b.get_value(),
                                   cnn.L1.w.get_value(),
                                   cnn.L1.b.get_value(),#])
                                   cnn.L2.w.get_value(),
                                   cnn.L2.b.get_value(),#])
                                   cnn.L3.w.get_value(),
                                   cnn.L3.b.get_value(),#])
                                   cnn.L4.w.get_value(),
                                   cnn.L4.b.get_value()])


  end_time = timeit.default_timer()
  print('Total time: %.2f' % ((end_time-start_time)/60.))


def draw_grid(img,img_size,grid_num):
  t=img_size/grid_num
  for i in range(grid_num):
    cv2.line(img,(t*(i+1),0),(t*(i+1),img_size-1),(255,255,255),1)
    cv2.line(img,(0,t*(i+1)),(img_size-1,t*(i+1)),(255,255,255),1)

def test_mlp(bs,nu,lr,fs,kernel,pool,bm,ep,l1,l2,wd,img_s,grid_s,filename,testfile):
  ##########################
  #       Load Data        #
  ##########################
  img_size = img_s
  channel_size = kernel[0]
  grid_size = grid_s
  filenameList, trainData, trainLabels = loadData(filename,grid_size,img_size)
  test_filenameList, testData, testLabels = loadData(testfile,grid_size,img_size)

#  for i in trainData[0]:
#    if i>0:
#      print i
#  print trainData[0].shape
#  raw_input()

##########################
#       Variable         #
##########################

  x = T.matrix('x')
  y_hat = T.matrix('y_hat')
  batch_size = bs
  epoch_num = ep
  neuron = nu
  learning_rate = lr
  filter_size = fs
  lambda1 = l1
  lambda2 = l2
  weight_decay = wd
  kernel=kernel
  poolFlag=pool
  border_mode=bm
  output_total = grid_size**2
  cnn_output_size = output_total*kernel[-1]

  cnn = CNN_MLP(x,filter_size,img_size,kernel,batch_size,poolFlag,border_mode)

  dnn_input = cnn.output.reshape((batch_size,cnn_output_size))
  dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,output_total,batch_size)

  params = cnn.params + dnn.params
  #L1 = ( abs(cnn.w).sum() + abs(dnn.L1.w).sum() + abs(dnn.L2.w).sum() )
  #L2 = ( (cnn.w**2).sum() + (dnn.L1.w**2).sum() + (dnn.L2.w**2).sum() )
  cost = ED(dnn.output,y_hat)
  #cost = OBJ(dnn.output,y_hat)
#  cost = YOLO(dnn.output,y_hat,batch_size,grid_size,class_num)
#  cost = ED(y_hat,dnn.output) + lambda1 * L1 + lambda2 * L2
#  cost = NLL(dnn.output,y_hat) + lambda1 * L1 + lambda2 * L2
  gparams = [ T.grad(cost,para) for para in params]
  g=theano.function(inputs=[x,y_hat],
                    outputs=[dnn.output,cost],
                    updates=MyUpdate(params,gparams,learning_rate,weight_decay))
  v=theano.function(inputs=[x],outputs=[dnn.output])

##########################
#    Training Model      #
##########################
  ans,c = g(trainData[0:1],trainLabels[0:1])
  print 'Test begin: [' + str(c) + ']'
  trainNetwork(g,v,trainData,trainLabels,batch_size,epoch_num,img_size,grid_size,dnn,cnn,testData,testLabels,filenameList,test_filenameList,filename)


def trail_test(bs,nu,lr,fs,kernel,pool,bm,ep,l1,l2,wd,img_s,grid_s,testfile,num):
  ##########################
  #       Load Data        #
  ##########################
  img_size = img_s
  channel_size = kernel[0]
  grid_size = grid_s
  test_filenameList, testData, testLabels = loadData(testfile,grid_size,img_size)

##########################
#       Variable         #
##########################

  x = T.matrix('x')
  y_hat = T.matrix('y_hat')
  batch_size = bs
  epoch_num = ep
  neuron = nu
  learning_rate = lr
  filter_size = fs
  lambda1 = l1
  lambda2 = l2
  weight_decay = wd
  kernel=kernel
  poolFlag=pool
  border_mode=bm
  output_total = grid_size**2
  cnn_output_size = output_total*kernel[-1]

  cnn = CNN_MLP(x,filter_size,img_size,kernel,batch_size,poolFlag,border_mode)

  dnn_input = cnn.output.reshape((batch_size,cnn_output_size))
  dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,output_total,batch_size)

  params = cnn.params + dnn.params
  cost = ED(dnn.output,y_hat)
  #cost = OBJ(dnn.output,y_hat)
  gparams = [ T.grad(cost,para) for para in params]
  g=theano.function(inputs=[x],outputs=[dnn.output])

  print 'Load w and b...'
  path='para/oracleTrain_org/'+str(num)+'/oracleTrain_org_para'
  save_file = open(path)
  dnn.L1.w.set_value(cPickle.load(save_file))
  dnn.L1.b.set_value(cPickle.load(save_file))
  dnn.L2.w.set_value(cPickle.load(save_file))
  dnn.L2.b.set_value(cPickle.load(save_file))
  cnn.L1.w.set_value(cPickle.load(save_file))
  cnn.L1.b.set_value(cPickle.load(save_file))
  cnn.L2.w.set_value(cPickle.load(save_file))
  cnn.L2.b.set_value(cPickle.load(save_file))
  cnn.L3.w.set_value(cPickle.load(save_file))
  cnn.L3.b.set_value(cPickle.load(save_file))
  cnn.L4.w.set_value(cPickle.load(save_file))
  cnn.L4.b.set_value(cPickle.load(save_file))
  save_file.close()

  ctr=0
  table=np.zeros((10,10,4))
  wrong_img=[]
  for i in range(testData.shape[0]):
    output = g(testData[i:i+1])
    predict = (output[0]>0.5)
    answer = (testLabels[i:i+1]==1)
    _x,_y = test_filenameList[i][16:-4].split('_')
    _x = int(_x)/(160/grid_size) #img_size is not the real size
    _y = int(_y)/(160/grid_size)
#    print test_filenameList[i]
#    print _x,_y
#    raw_input()
#    print testLabels[i:i+1].shape
#    print predict.shape
#    print answer.shape
#    raw_input()
    #print predict.sum(),answer.sum()
    if (predict==answer).all():
      #print str(test_filenameList[i])+' Correct'
      table[_y,_x,0]+=1
      ctr+=1
    else:
      #print str(test_filenameList[i])+' Wrong'
      table[_y,_x,1]+=1
      wrong_img+=[i]
  table[:,:,2]=table[:,:,0]*100/(table[:,:,0]+table[:,:,1])
  table[:,:,3]=table[:,:,1]*100/(table[:,:,0]+table[:,:,1])
  print("%.3f%%"%(ctr*100./testLabels.shape[0]))
  print tabulate(table[:,:,3],tablefmt="grid")

  # Each image Check
  for i in wrong_img:
    TF=0;FT=0
    TF_buffer=[]
    FT_buffer=[]
    img = cv2.imread(test_filenameList[i])
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    output = g(testData[i:i+1])[0]
    predict = (output[0]>0.5)
    answer = (testLabels[i:i+1]==1)
    total_true = answer.sum()
    total_false = (~predict==answer).sum()
    _, wrong_index = np.where(~predict==answer)
    for j in wrong_index:
      if predict[j] and ~answer[0,j]:
        FT+=1
        FT_buffer+=[(j%grid_size,j/grid_size)]
      elif ~predict[j] and answer[0,j]:
        TF+=1
        TF_buffer+=[(j%grid_size,j/grid_size)]
#    print wrong_index
#    print predict
#    print answer
#    print total_false
#    raw_input()
    print test_filenameList[i]
    print 'Answer'
    print testLabels[i].reshape((10,10))
    print 'Predict'
    print output.reshape((10,10))
    print 'Total True: '+str(total_true)
    print 'Total False: '+str(total_false )
    print 'Fact True => Predict False: '+str(TF)+' '+str(TF_buffer)
    print 'Fact False => Predict True: '+str(FT)+' '+str(FT_buffer)

    img2 = cv2.resize(img,(img_size,img_size))
    for i in TF_buffer:
      print i
      print img2[i[1]*2:(i[1]+1)*2,i[0]*2:(i[0]+1)*2,0]

    img = cv2.resize(img,(160,160))
    draw_grid(img,160,grid_size)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
# batch, neuron, lr, filter,kernel,pool,border_mode l1,l2,wd, img, grid,
#  test_mlp(1,512,0.01,[3,3,3,3],#filter_size
#           [3,3,3,3,3],#kernel
#          [False,True,False,False,False],#pool
#          [True,False,True,False],#border_mode
#          2000,0,0,0,20,10,'oracleTrain_org','oracleTest')
  trail_test(1,512,0.01,[3,3,3,3],#filter_size
           [3,3,3,3,3],#kernel_size
          [False,True,False,False,False],#pool True: pooling
          [True,False,True,False],#border_mode True: Padding
          2000,0,0,0,20,10,'oracleTest',9741)
  pass
