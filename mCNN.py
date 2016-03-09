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
  def __init__(self, input, filter_shape, image_shape, poolsize=(2,2)):
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
    conv_out = conv.conv2d(input,self.w)
    output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
#    output = T.nnet.relu(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
    pool_out = downsample.max_pool_2d(output, poolsize, ignore_border=True)
    self.output = pool_out
    self.params = [self.w, self.b]

##########################
#          MLP           #
##########################
class MLP(object):
  def __init__(self,input,y_hat,n_in,n_hidden,n_out,n_batch):
    self.L1 = HiddenLayer(input,n_in,n_hidden,n_batch,Sigmoid)
    self.L2 = HiddenLayer(self.L1.output,n_hidden,n_hidden,n_batch,Sigmoid)
    self.L3 = HiddenLayer(self.L2.output,n_hidden,n_out,n_batch,Sigmoid)
    self.params = self.L1.params + self.L2.params + self.L3.params
    self.output = self.L3.output

##########################
#      Load Data         #
##########################
def loadData(filename,grid_num,class_num,img_size):
  #trainLabels(training_num, grid_sq, xywhC)
  #trainData will follow img_label to build the trainData
  label = np.loadtxt(filename,dtype='str')
  grid_sq = grid_num **2
  trainData = []
  trainLabels = []
  filenameList = []
  #print label; print label.shape
  for i in range(label.shape[0]):
    #img = cv2.imread(os.path.join('data/',label[i][0]))
    img = cv2.imread(label[i][0])
    img = cv2.resize(img,(img_size,img_size))
    filenameList+=[label[i][0]]
    #cv2.imshow('img',img)
    #cv2.waitKey(0) & 0xFF
    #cv2.destroyAllWindows()
    xywh = np.repeat(label[i][1:5].reshape((1,4)),grid_sq,axis=0)
    xywh = xywh.astype(np.float)
    #print xywh, xywh.shape
    clas = np.repeat(label[i][5+grid_sq:].reshape((1,class_num)),grid_sq,axis=0)
    clas = clas.astype(np.int)
    #print clas, clas.shape

    score = label[i][5:5+grid_sq].reshape((grid_sq,1))
    score = score.astype(np.float)
    tmp = np.concatenate((xywh,score,clas),axis=1)
    #print tmp, tmp.shape
    #print img.shape
    trainData += [img.reshape((3*img_size**2))]
    trainLabels += [tmp.reshape((grid_sq*(5+class_num)))]
  trainData = np.asarray(trainData)
  trainLabels = np.asarray(trainLabels)#.reshape((label.shape[0],grid_sq,5+class_num))
  #print trainLabels, trainLabels.shape
  #print trainData, trainData.shape
  return filenameList,trainData, trainLabels

##########################
# Train & Valid function #
##########################
def printScore(output,answer,img_size,grid_size,class_num,epoch,index):
  output = output[0]
  gird_sq = grid_size**2
  scoreCtr = 0
  logname='log'
  title = '\n********************* Epoch: '+str(epoch)+', Index: '+str(index)+' *********************\n'
  record(logname,3,title)
#  print 'output '+str(output.shape)
#  print 'answer '+str(answer.shape)
  pre_score=output[0,4::5+class_num].reshape((grid_size,grid_size))
  cor_score=answer[0,4::5+class_num].reshape((grid_size,grid_size))

  if pre_score.argmax()==cor_score.argmax():
    scoreCtr += 1
  record(logname,3, "########### Object Score###########")
  record(logname,3, "Predict")
  record(logname,3, tabulate(pre_score,tablefmt='grid'))
  record(logname,3, "Answer")
  record(logname,3, tabulate(cor_score,tablefmt='grid'))
  record(logname,3, "###################################")

  pre_x=output[0,0::5+class_num].reshape((grid_size,grid_size))
  cor_x=answer[0,0::5+class_num].reshape((grid_size,grid_size))
  pre_y=output[0,1::5+class_num].reshape((grid_size,grid_size))
  cor_y=answer[0,1::5+class_num].reshape((grid_size,grid_size))
  pre_w=output[0,2::5+class_num].reshape((grid_size,grid_size))
  cor_w=answer[0,2::5+class_num].reshape((grid_size,grid_size))
  pre_h=output[0,3::5+class_num].reshape((grid_size,grid_size))
  cor_h=answer[0,3::5+class_num].reshape((grid_size,grid_size))

  pre_gridClass=[]
  cor_gridClass=[]
  for i in range(gird_sq):
    pre_gridClass+=[output[0,5+(5+class_num)*i:5+(5+class_num)*i+class_num]]
    cor_gridClass+=[answer[0,5+(5+class_num)*i:5+(5+class_num)*i+class_num]]
  pre_gridClass = np.asarray(pre_gridClass).reshape((grid_size,grid_size,class_num))
  cor_gridClass = np.asarray(cor_gridClass).reshape((grid_size,grid_size,class_num))

  pre_list=[]
  cor_list=[]
  record(logname,3, "############## BBox ###############")
  for i in range(grid_size):
    for j in range(grid_size):
      if cor_score[i][j]>0:
        delta=0
        delta+=abs(cor_x[i][j]-pre_x[i][j])
        delta+=abs(cor_y[i][j]-pre_y[i][j])
        delta+=abs(cor_w[i][j]-pre_w[i][j])
        delta+=abs(cor_h[i][j]-pre_h[i][j])
#        print("BBox center:\t(%.2f,%.2f,%.2f,%.2f)" %(cor_x[i][j],cor_y[i][j],
#                                                    cor_w[i][j],cor_h[i][j]))
#        print("Grid(%d,%d):\t(%.2f,%.2f,%.2f,%.2f), delta: %.2f" %(i,j,pre_x[i][j],pre_y[i][j],
#                                                                pre_w[i][j],pre_h[i][j],delta))
        record(logname,3,"BBox Center:")
        record(logname,3,str([round(cor_x[i][j],2),
                              round(cor_y[i][j],2),
                              round(cor_w[i][j],2),
                              round(cor_h[i][j],2)]))
        grid_xy = 'Grid('+str(i)+','+str(j)+'): '
        record(logname,3,grid_xy)
        record(logname,3,str([round(pre_x[i][j],2),
                              round(pre_y[i][j],2),
                              round(pre_w[i][j],2),
                              round(pre_h[i][j],2),
                              round(delta,2)]))
        pre_list+=[pre_gridClass[i,j,0]]#.argmax()]
        cor_list+=[cor_gridClass[i,j,:].argmax()]
      else:
        pre_list+=[-1]
        cor_list+=[-1]
  record(logname,3, "###################################")
  record(logname,3, "############# Class ###############")
  pre_list = np.asarray(pre_list).reshape((grid_size,grid_size))
  cor_list = np.asarray(cor_list).reshape((grid_size,grid_size))
  record(logname,3, "Predict")
  record(logname,3, tabulate(pre_list,tablefmt="grid"))
  record(logname,3, "Answer")
  record(logname,3, tabulate(cor_list,tablefmt="grid"))
  record(logname,3, "###################################")

  return scoreCtr,delta

def trainNetwork(g,v,trainData,trainLabels,batch_size,epoch_num,img_size,grid_size,class_num,dnn,cnn1,cnn2,cnn3,testData,testLabels):
  good_scoreCtr = 0
  good_accDelta = 99999
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
      trainAccDelta = 0
      testScoreCtr = 0
      testAccDelta = 0
      for x in range(trainLabels.shape[0]):
        output = v(trainData[x:x+1])
        tmp, delta = printScore(output,trainLabels[x:x+1],img_size,grid_size,class_num,e+1,x+1)
        trainScoreCtr += tmp
        trainAccDelta += delta
      for x in range(testLabels.shape[0]):
        output = v(testData[x:x+1])
        tmp, delta = printScore(output,testLabels[x:x+1],img_size,grid_size,class_num,e+1,x+1)
        testScoreCtr += tmp
        testAccDelta += delta
      print trainScoreCtr, good_scoreCtr, trainAccDelta, good_accDelta
      print testScoreCtr, good_scoreCtr, testAccDelta, good_accDelta
      if (trainScoreCtr+testScoreCtr) >= good_scoreCtr and (trainAccDelta+testAccDelta) < good_accDelta:
        print "SAVE PARAMETERS!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        good_scoreCtr = trainScoreCtr+testScoreCtr
        good_accDelta = trainAccDelta+10*testAccDelta
        save_params('para',params=[dnn.L1.w.get_value(),
                                      dnn.L1.b.get_value(),
                                      dnn.L2.w.get_value(),
                                      dnn.L2.b.get_value(),
                                      cnn1.w.get_value(),
                                      cnn1.b.get_value(),
                                      cnn2.w.get_value(),
                                      cnn2.b.get_value(),
                                      cnn3.w.get_value(),
                                      cnn3.b.get_value()])


  end_time = timeit.default_timer()
  print('Total time: %.2f' % ((end_time-start_time)/60.))

def show(answer,filename,img_size,grid_size,class_num,real_answer):
#  print img_size,grid_size
#  print filename
  grid_sq = grid_size **2
  answer = answer[0].reshape((grid_sq,5+class_num))
  real_answer = real_answer[0].reshape((grid_sq,5+class_num))
#  print answer.shape
#  raw_input()
  #img = cv2.imread(os.path.join('data/',filename))
  img = cv2.imread(filename)
  img = cv2.resize(img, (img_size,img_size))
  for i in range(grid_sq):
    print answer[i]
    if real_answer[i,4]>0.0:#*answer[i,5]>0.1:
      center_x = int((i%grid_size+answer[i,0])*(img_size/grid_size))
      center_y = int((i/grid_size+answer[i,1])*(img_size/grid_size))
      real_w = int(answer[i,2]*(img_size))
      real_h = int(answer[i,3]*(img_size))
      start_x = int((2*center_x+real_w-1)/2)
      start_y = int((2*center_y+real_h-1)/2)
      end_x = int((2*center_x-real_w+1)/2)
      end_y = int((2*center_y-real_h+1)/2)
      print start_x,start_y,end_x,end_y
      cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(255,0,0),2)
    if answer[i,4]>0.03:#*answer[i,5]>0.1:
      center_x = int((i%grid_size+answer[i,0])*(img_size/grid_size))
      center_y = int((i/grid_size+answer[i,1])*(img_size/grid_size))
      real_w = int(answer[i,2]*(img_size))
      real_h = int(answer[i,3]*(img_size))
      start_x = int((2*center_x+real_w-1)/2)
      start_y = int((2*center_y+real_h-1)/2)
      end_x = int((2*center_x-real_w+1)/2)
      end_y = int((2*center_y-real_h+1)/2)
      print start_x,start_y,end_x,end_y
      cv2.rectangle(img,(start_x,start_y),(end_x,end_y),(0,255,0),1)
#      cv2.circle(img,(center_y,center_x),min(real_w,real_h),(255,0,0),3)
  plt.imshow(img)
  plt.show()

def test_mlp(bs,nu,lr,fs,ep,l1,l2,wd,img_s,chl_s,grid_s,cls_n,filename):
  ##########################
  #       Load Data        #
  ##########################
  img_size = img_s
#      cv2.circle(img,(center_y,center_x),min(real_w,real_h),(255,0,0),3)
  plt.imshow(img)
  plt.show()

def test_mlp(bs,nu,lr,fs,ep,l1,l2,wd,img_s,chl_s,grid_s,cls_n,filename,testfile):
  ##########################
  #       Load Data        #
  ##########################
  img_size = img_s
  channel = chl_s
  grid_size = grid_s
  class_num = cls_n
  filenameList, trainData, trainLabels = loadData(filename,grid_size,
  class_num,img_size)
  test_filenameList, testData, testLabels = loadData(testfile,grid_size,
  class_num,img_size)


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
  output_total = (5+class_num)*grid_size**2
  kernel=[channel,10,20,50]

  cnn1_input = x.reshape((batch_size,kernel[0],img_size,img_size))
  cnn1 = CNN_Layer(cnn1_input,
                  filter_shape=(kernel[1],kernel[0],filter_size,filter_size),
                  image_shape=(batch_size,kernel[0],img_size,img_size),
                  poolsize=(2,2))
  tmp_size = (img_size - filter_size + 1)/2
  cnn2 = CNN_Layer(cnn1.output,
                  filter_shape=(kernel[2],kernel[1],filter_size,filter_size),
                  image_shape=(batch_size,kernel[1],tmp_size,tmp_size),
                  poolsize=(2,2))
#  tmp_size = (tmp_size - filter_size + 1)/2
#  cnn3 = CNN_Layer(cnn2.output,
#                  filter_shape=(kernel[3],kernel[2],filter_size,filter_size),
#                  image_shape=(batch_size,kernel[2],tmp_size,tmp_size),
#                  poolsize=(2,2))

  tmp_size = (tmp_size - filter_size + 1)/2
  cnn_output_size = kernel[2]*tmp_size**2
  dnn_input = cnn2.output.reshape((batch_size,cnn_output_size))
  dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,output_total,batch_size)

  params = cnn1.params + cnn2.params + dnn.params #cnn3.params + dnn.params
#  L1 = ( abs(cnn1.w).sum() + abs(cnn2.w).sum() + abs(cnn3.w).sum() + abs(dnn.L1.w).sum() + abs(dnn.L2.w).sum() )
#  L2 = ( (cnn1.w**2).sum() + (cnn2.w**2).sum() + (cnn3.w**2).sum() + (dnn.L1.w**2).sum() + (dnn.L2.w**2).sum() )
  cost = YOLO(dnn.output,y_hat,batch_size,grid_size,class_num)
#  cost = ED(y_hat,dnn.output) + lambda1 * L1 + lambda2 * L2
#  cost = NLL(dnn.output,y_hat) + lambda1 * L1 + lambda2 * L2
  gparams = [ T.grad(cost,para) for para in params]
  g=theano.function(inputs=[x,y_hat],
                    outputs=[dnn.output,cost],
                    updates=MyUpdate(params,gparams,learning_rate,weight_decay))
  v=theano.function(inputs=[x],outputs=[dnn.output])

#  t=theano.function(inputs=[x],
#                    outputs=[cnn1.output])
#  ans = t(trainData[0:1])
#  print ans[0].shape
##########################
#    Training Model      #
##########################
  ans,c = g(trainData[0:1],trainLabels[0:1])
  print 'Test begin: [' + str(c) + ']'

  trainNetwork(g,v,trainData,trainLabels,batch_size,epoch_num,img_size,grid_size,class_num,dnn,cnn1,cnn2,cnn2,testData,testLabels)

def trail_test(bs,nu,lr,fs,img_s,chl_s,grid_s,cls_n,filename):
  #load image
  img_size = img_s
  channel = chl_s
  grid_size = grid_s
  class_num = cls_n
  filenameList, trainData, trainLabels = loadData(filename,grid_size,
  class_num,img_size)

  x = T.matrix('x')
  y_hat = T.matrix('y_hat')
  batch_size = bs
#  epoch_num = ep
  neuron = nu
  learning_rate = lr
  filter_size = fs
  output_total = (5+class_num)*grid_size**2
  kernel=[channel,10,20,50]

  cnn1_input = x.reshape((batch_size,kernel[0],img_size,img_size))
  cnn1 = CNN_Layer(cnn1_input,
                  filter_shape=(kernel[1],kernel[0],filter_size,filter_size),
                  image_shape=(batch_size,kernel[0],img_size,img_size),
                  poolsize=(2,2))
  tmp_size = (img_size - filter_size + 1)/2
  cnn2 = CNN_Layer(cnn1.output,
                  filter_shape=(kernel[2],kernel[1],filter_size,filter_size),
                  image_shape=(batch_size,kernel[1],tmp_size,tmp_size),
                  poolsize=(2,2))
  tmp_size = (tmp_size - filter_size + 1)/2
  cnn3 = CNN_Layer(cnn2.output,
                  filter_shape=(kernel[3],kernel[2],filter_size,filter_size),
                  image_shape=(batch_size,kernel[2],tmp_size,tmp_size),
                  poolsize=(2,2))

  tmp_size = (tmp_size - filter_size + 1)/2
  cnn_output_size = kernel[3]*tmp_size**2
  dnn_input = cnn3.output.reshape((batch_size,cnn_output_size))
  dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,output_total,batch_size)

  params = cnn1.params + cnn2.params + cnn3.params + dnn.params
  g=theano.function(inputs=[x],outputs=[dnn.output])

  print 'Load w and b...'
  save_file = open('para')
  dnn.L1.w.set_value(cPickle.load(save_file))
  dnn.L1.b.set_value(cPickle.load(save_file))
  dnn.L2.w.set_value(cPickle.load(save_file))
  dnn.L2.b.set_value(cPickle.load(save_file))
  cnn1.w.set_value(cPickle.load(save_file))
  cnn1.b.set_value(cPickle.load(save_file))
  cnn2.w.set_value(cPickle.load(save_file))
  cnn2.b.set_value(cPickle.load(save_file))
  cnn3.w.set_value(cPickle.load(save_file))
  cnn3.b.set_value(cPickle.load(save_file))
  save_file.close()

  #Test and print
  for i in range(trainData.shape[0]):
    y = g(trainData[i*batch_size:(i+1)*batch_size])
    print trainLabels[i*batch_size:(i+1)*batch_size].shape
    show(trainLabels[i*batch_size:(i+1)*batch_size],filenameList[i],img_size,grid_size,class_num,trainLabels[i*batch_size:(i+1)*batch_size])
    show(y,filenameList[i],img_size,grid_size,class_num,trainLabels[i*batch_size:(i+1)*batch_size])
    #raw_input()

if __name__ == '__main__':
# batch, neuron, lr, filter, l1,l2,wd, img,channel, grid, classNum
  test_mlp(1,1024,0.0001,5,300,0,0,0,156,3,4,2,'4grid-train','4grid-test')
#  trail_test(1,1024,0.0000001,5,156,3,4,2,'4grid-test')
  pass
