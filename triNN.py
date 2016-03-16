#--------------------------------------------------------------------------
#
#  Project Nmae: OOD.py
#  Goal: Build a single neural network to do object detection & classify the image
#
#
#  Date: 2016.02.23
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
  y = T.maximum(0,x)
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
  def __init__(self, input, filter_shape, image_shape, poolsize=(2,2), poolFlag=False):
    self.input = input
    fan_in = np.prod(filter_shape[1:])
    fan_out = np.prod(filter_shape[0]*np.prod(filter_shape[2:]/np.prod(poolsize)))

    w_bound = np.sqrt(6./(fan_in+fan_out))
    w_values = np.asarray(np.random.uniform(-w_bound,w_bound,size=filter_shape))#,dtype=theano.config.floatX)
    w = theano.shared(value=w_values,name='W')
    b_values = np.asarray(np.zeros((filter_shape[0],)))#,dtype=theano.config.floatX)
    b = theano.shared(value=b_values,name='B')
    self.w = w
    self.b = b
    conv_out = conv.conv2d(input,self.w)
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
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
    self.params = self.L1.params + self.L2.params# + self.L3.params
    self.output = self.L2.output

##########################
#        CNN-MLP         #
##########################
class CNN_MLP(object):
  def __init__(self, input, filter_size, img_size, kernel, batch_size,poolFlag):
    self.input = input.reshape((batch_size,kernel[0],img_size,img_size))
    self.L1 = CNN_Layer(self.input,
                  filter_shape=(kernel[1],kernel[0],filter_size[0],filter_size[0]),
                  image_shape=(batch_size,kernel[0],img_size,img_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[0])
    if poolFlag[0]:
      tmp_size = (img_size - filter_size[0] + 1)/2
    else:
      tmp_size = (img_size - filter_size[0] + 1)

    self.L2 = CNN_Layer(self.L1.output,
                  filter_shape=(kernel[2],kernel[1],filter_size[1],filter_size[1]),
                  image_shape=(batch_size,kernel[1],tmp_size,tmp_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[1])
    if poolFlag[1]:
      tmp_size = (tmp_size - filter_size[1] + 1)/2
    else:
      tmp_size = (tmp_size - filter_size[1] + 1)

    self.L3 = CNN_Layer(self.L2.output,
                  filter_shape=(kernel[3],kernel[2],filter_size[1],filter_size[1]),
                  image_shape=(batch_size,kernel[2],tmp_size,tmp_size),
                  poolsize=(2,2),
                  poolFlag=poolFlag[2])
    if poolFlag[2]:
      tmp_size = (tmp_size - filter_size[2] + 1)/2
    else:
      tmp_size = (tmp_size - filter_size[2] + 1)

#    self.L4 = CNN_Layer(self.L3.output,
#                  filter_shape=(kernel[4],kernel[3],filter_size[1],filter_size[1]),
#                  image_shape=(batch_size,kernel[3],tmp_size,tmp_size),
#                  poolsize=(2,2),
#                  poolFlag=poolFlag[3])
#    if poolFlag[3]:
#      tmp_size = (tmp_size - filter_size[3] + 1)/2
#    else:
#      tmp_size = (tmp_size - filter_size[3] + 1)


    self.output_size = kernel[3]*tmp_size**2
    self.params = self.L1.params + self.L2.params + self.L3.params# + self.L4.params
    self.output = self.L3.output.reshape((batch_size,self.output_size))

def printOutput(last_params,trainData,testData,layer_num,filter_size,img_size,batch_size,kernel,poolFlag,label):
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
                    poolFlag=poolFlag[i])
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



def test_mlp(bs,nu,lr,fs,ep,l1,l2,wd,img_s,chl_s,grid_s,cls_n,filename,testfile):
  ##########################
  #       Load Data        #
  ##########################
  img_size = img_s
  channel = chl_s
  grid_size = grid_s
  class_num = cls_n
  train = np.loadtxt(filename)
  test = np.loadtxt(testfile)
#  print train.shape
#  print test.shape
  trainData, trainLabels= train[:,0:-16],train[:,-16:]
  testData, testLabels= test[:,0:-16],test[:,-16:]
  label = np.concatenate([trainLabels,testLabels],axis=0)
#  print trainData.shape, trainLabels.shape
#  print testData.shape, testLabels.shape
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
  output_total = grid_size**2
  kernel=[channel,1,1,1]
  poolFlag=[True,False,False]

  cnn = CNN_MLP(x,filter_size,img_size,kernel,batch_size,poolFlag)
  params = cnn.params
#  cost = ED(cnn.output,y_hat)
  cost = OBJ(cnn.output,y_hat)
  gparams = [ T.grad(cost,para) for para in params]
  g=theano.function(inputs=[x,y_hat],
                    outputs=[cnn.output,cost],
                    updates=MyUpdate(params,gparams,learning_rate,weight_decay))
  v=theano.function(inputs=[x,y_hat],outputs=[cnn.output,cost])
##########################
#    Training Model      #
##########################
  last_params=[]
  last_params+=[cnn.L1.w.get_value(),
                cnn.L1.b.get_value(),
                cnn.L2.w.get_value(),
                cnn.L2.b.get_value(),
                cnn.L3.w.get_value(),
                cnn.L3.b.get_value()]
#                cnn.L4.w.get_value(),
#                cnn.L4.b.get_value()]
  for e in range(epoch_num):
    for i in range(trainData.shape[0]):
      ans,cost = g(trainData[i:i+1],trainLabels[i:i+1])
      print("Epoch %d, cost: %.3f"%(e+1,cost))
    if (e+1)%10==0:
      #train
      trainCtr=0
      for j in range(trainData.shape[0]):
        tmp,cost = v(trainData[j:j+1],trainLabels[j:j+1])
        if tmp[0].argmax() == trainLabels[j].argmax():
          trainCtr+=1
      #test
      testCtr=0
      for k in range(testData.shape[0]):
        tmp,cost = v(testData[k:k+1],trainLabels[k:k+1])
#        print tmp[0]
#        print testLabels[k]
#        print cost
        if tmp[0].argmax()==testLabels[k].argmax():
          testCtr+=1
      print("Train: %d; Test: %d\n"%(trainCtr,testCtr))
      key=raw_input("Enter x to check the delta of W: ")
      if key=='x':
#        print "SHOW Delta_W"
#        print (cnn.L1.w.get_value()-last_params[-8])/(last_params[-8])*100
#        print (cnn.L2.w.get_value()-last_params[-6])/(last_params[-6])*100
#        print (cnn.L3.w.get_value()-last_params[-4])/(last_params[-4])*100
#        print (cnn.L4.w.get_value()-last_params[-2])/(last_params[-2])*100
#        print "SHOW Delta_W"
#        raw_input()
        printOutput(last_params,trainData,testData,3,filter_size,img_size,batch_size,kernel,poolFlag,label)
      last_params += [cnn.L1.w.get_value(),
                      cnn.L1.b.get_value(),
                      cnn.L2.w.get_value(),
                      cnn.L2.b.get_value(),
                      cnn.L3.w.get_value(),
                      cnn.L3.b.get_value()]
#                      cnn.L4.w.get_value(),
#                      cnn.L4.b.get_value()]

if __name__ == '__main__':
  #def test_mlp(bs,nu,lr,fs,ep,l1,l2,wd,img_s,chl_s,grid_s,cls_n,filename,testfile):
  # Variable
  test_mlp(1,256,0.09,[5,3,3],100000,0,0,0,20,1,4,0,'train','test')
  pass
