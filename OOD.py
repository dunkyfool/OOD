#--------------------------------------------------------------------------
#
#  Project Nmae: mlp.py
#  Function: Modelize MULTI-LAYER Neural Network
#
#
#  Date: 2015.12.25
#  Author: Jackie
#--------------------------------------------------------------------------
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

##########################
#   Actvation Function   #
##########################
def ReLU(x):
  y = T.maximum(0.0,x)
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
    pool_out = downsample.max_pool_2d(output, poolsize, ignore_border=True)
    self.output = pool_out
    self.params = [self.w, self.b]

##########################
#          MLP           #
##########################
class MLP(object):
  def __init__(self,input,y_hat,n_in,n_hidden,n_out,n_batch):
    self.L1 = HiddenLayer(input,n_in,n_hidden,n_batch,Sigmoid)
    self.L2 = HiddenLayer(self.L1.output,n_hidden,n_out,n_batch,Softmax)
    self.params = self.L1.params + self.L2.params
    self.output = self.L2.output

def loadData(filename,grid_num,class_num):
  #trainLabels(training_num, grid_sq, xywhC)
  label = np.loadtxt(filename,dtype='str')
  grid_sq = grid_num **2
  trainLabels = []
  #print label; print label.shape
  for i in range(label.shape[0]):
    xywh = np.repeat(label[i][1:5].reshape((1,4)),grid_sq,axis=0)
    #print xywh, xywh.shape
    clas = np.repeat(label[i][5+grid_sq:].reshape((1,class_num)),grid_sq,axis=0)
    #print clas, clas.shape
    tmp = np.concatenate((xywh,label[i][5:5+grid_sq].reshape((grid_sq,1)),clas),axis=1)
    #print tmp, tmp.shape
    trainLabels += [tmp]
  trainLabels = np.asarray(trainLabels).reshape((label.shape[0],grid_sq,5+class_num))
  print trainLabels, trainLabels.shape

def test_mlp(bs,nu,lr,fs,ep,l1,l2,wd):
  ##########################
  #       Load Data        #
  ##########################
  loadData('img_label',4,2)


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
  cnn_output_size = pow((image_size-filter_size+1)/2,2)
  good_record = 0

  cnn_input = x.reshape((batch_size,1,image_size,image_size))
  cnn = CNN_Layer(cnn_input,
                  filter_shape=(1,1,filter_size,filter_size),
                  image_shape=(batch_size,1,image_size,image_size),
                  poolsize=(2,2))

  dnn_input = cnn.output.reshape((batch_size,cnn_output_size))
  dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,101,batch_size)

  params = cnn.params + dnn.params
  L1 = ( abs(cnn.w).sum() + abs(dnn.L1.w).sum() + abs(dnn.L2.w).sum() )
  L2 = ( (cnn.w**2).sum() + (dnn.L1.w**2).sum() + (dnn.L2.w**2).sum() )
  #L1 = T.sum(map(abs,params))
  #L2 = T.sum(map(lambda x: x ** 2,params))
#  cost = ED(y_hat,dnn.output) + lambda1 * L1 + lambda2 * L2
  cost = NLL(dnn.output,y_hat) + lambda1 * L1 + lambda2 * L2
  gparams = [ T.grad(cost,para) for para in params]
  g=theano.function(inputs=[x,y_hat],
                    outputs=[dnn.output,cost],
                    updates=MyUpdate(params,gparams,learning_rate,weight_decay))
  valid_model=theano.function(inputs=[x],outputs=[dnn.output])

##########################
#    Training Model      #
##########################

  print "start testing"
  start_time = timeit.default_timer()
  for e in range(epoch_num):
    for i in range(trainData.shape[0]/batch_size):
      y,c = g(trainData[i*batch_size:(i+1)*batch_size],trainLabel[i*batch_size:(i+1)*batch_size])
      # print("Epoch=%d, Progress: %.2f%%, cost=%.2f" %(e+1, i*100./(trainData.shape[0]/batch_size),c))
    shuffle(trainData,trainLabel,trainData.shape[1])
    ### Validation on EVERY FIVE EPOCHS [START]
    if (e+1)%10 == 0:
        dis_trainLabels_correct=np.zeros(101)
        dis_testLabels_correct=np.zeros(101)
        trainCorrect=0
        trainCtr=0
        for k in range(trainData.shape[0]/batch_size):
          trainY1 = valid_model(trainData[k*batch_size:(k+1)*batch_size])
          for j in range(batch_size):
            if np.argmax(trainLabel[trainCtr]) == np.argmax(trainY1[0][j]):
              trainCorrect+=1
              # Record each class's accuracy
              dis_trainLabels_correct[np.argmax(trainLabel[trainCtr])]+=1
            trainCtr+=1
        testCorrect=0
        testCtr=0
        testWrongClass=[[]]*101
        misRecognizedClass=[0]*101
        for k in range(testData.shape[0]/batch_size):
          testY1 = valid_model(testData[k*batch_size:(k+1)*batch_size])
          for j in range(batch_size):
            if np.argmax(testLabel[testCtr]) == np.argmax(testY1[0][j]):
              testCorrect+=1
              # Record each class's accuracy
              dis_testLabels_correct[np.argmax(testLabel[testCtr])]+=1
            else:
              # Record wrong test result
              correctLabel = np.argmax(testLabel[testCtr])
              wrongLabel = np.argmax(testY1[0][j])
              flag=0
              misRecognizedClass[wrongLabel] = misRecognizedClass[wrongLabel]+1
              if len(testWrongClass[correctLabel]) == 0:
                testWrongClass[correctLabel] = testWrongClass[correctLabel]+[[wrongLabel,1]]
              else:
                for b in range(len(testWrongClass[correctLabel])):
                  if testWrongClass[correctLabel][b][0] == wrongLabel:
                    testWrongClass[correctLabel][b][1]+=1
                    flag=1
                    break
                if flag == 0:
                  testWrongClass[correctLabel] = testWrongClass[correctLabel]+[[wrongLabel,1]]
            testCtr+=1

        print("e=%3d Total = %d,train Correct_rate: %.2f%% test Correct_rate: %.2f%%" %(e+1,trainCtr,trainCorrect*100./trainCtr,testCorrect*100./testCtr))

        record('20160111_auto_cnn_dnn_log',2,(e+1,trainCorrect*100./trainCtr,batch_size,neuron,learning_rate,filter_size,lambda1,lambda2,testCorrect*100./testCtr))

#        if trainCorrect*100./trainCtr > good_record:
#          good_record = trainCorrect*100./trainCtr
#          file_name = str(bs)+'_'+str(nu)+'_'+str(lr)+'_'+str(fs)+'_para'
#          save_params(file_name,
#                      params=[dnn.L1.w.get_value(),
#                              dnn.L1.b.get_value(),
#                              dnn.L2.w.get_value(),
#                              dnn.L2.b.get_value(),
#                              cnn.w.get_value(),
#                              cnn.b.get_value()])
### Validation on EVERY FIVE EPOCHS [END]
  end_time = timeit.default_timer()

  print('Total time: %.2f' % ((end_time-start_time)/60.))


# def trail_test(bs,nu,lr,fs):
#   #rebuild a new one
#   mat = sio.loadmat('caltech101_silhouettes_16')
#   train = mat['X']
#   ans = mat['Y'].reshape(train.shape[0])
#   mx = np.identity(101)
#   label=[]
#   for key in range(train.shape[0]):
#     label.append(mx[ans[key]-1])
#   label = np.array(label)

#   x = T.matrix('x')
#   y_hat = T.matrix('y_hat')
#   batch_size = bs
#   epoch_num = 20
#   neuron = nu
#   learning_rate = lr
#   filter_size = fs
#   image_size = 16
#   cnn_output_size = pow((image_size-filter_size+1)/2,2)

#   cnn_input = x.reshape((batch_size,1,image_size,image_size))
#   cnn = CNN_Layer(cnn_input,filter_shape=(1,1,filter_size,filter_size),
#                   image_shape=(batch_size,1,image_size,image_size),
#                   poolsize=(2,2))

#   dnn_input = cnn.output.reshape((batch_size,cnn_output_size))
#   dnn = MLP(dnn_input,y_hat,cnn_output_size,neuron,101,batch_size)

#   trail_model=theano.function(inputs=[x],outputs=[dnn.output])

#   print 'Load w and b...'
#   save_file = open('500_512_0.001_3_para')
#   dnn.L1.w.set_value(cPickle.load(save_file))
#   dnn.L1.b.set_value(cPickle.load(save_file))
#   dnn.L2.w.set_value(cPickle.load(save_file))
#   dnn.L2.b.set_value(cPickle.load(save_file))
#   cnn.w.set_value(cPickle.load(save_file))
#   cnn.b.set_value(cPickle.load(save_file))
#   save_file.close()

#   correct=0
#   ctr=0
#   for i in range(train.shape[0]/batch_size):
#     y1 = trail_model(train[i*batch_size:(i+1)*batch_size])
#     for j in range(batch_size):
#       if np.argmax(label[ctr])+1 == np.argmax(y1[0][j])+1:
#           correct+=1
#       ctr+=1

#   print("Total = %d, Correct_rate: %.2f%%" %(ctr,correct*100./ctr))


if __name__ == '__main__':
#  loadData('img_label',4,2)
  test_mlp(50,128,0.06,5,1000,0,0,0)
#  trail_test(1,512,0.001,3)
