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

def MyUpdate(parameters,gradients):
  parameters_updates = []
  for p,g in izip(parameters,gradients):
    parameters_updates = parameters_updates + [(p,p-(0.1)*g)]
  return parameters_updates

x = T.matrix('x')
y_hat = T.matrix('y_hat')
w = theano.shared(
             value=np.ones((28*28, 16*7)),
             name='w')

output = T.dot(x,w).reshape((16,7))
#cost = T.sum((output[:,4]-y_hat[:,4])**2)
x1 = ((output[:,4]-y_hat[:,4])**2)

x2 = np.repeat(x1.reshape((16,1)),4,axis=1)
x3 = np.repeat(x1.reshape((16,1)),2,axis=1)
delta_xywh = ((output[:,0:4]-y_hat[:,0:4])**2)
delta_class= ((output[:,5:]-y_hat[:,5:])**2)

#delta_x = (output[:,0]-y_hat[:,0])**2
#delta_y = (output[:,1]-y_hat[:,1])**2
#delta_w = (output[:,2]-y_hat[:,2])**2
#delta_h = (output[:,3]-y_hat[:,3])**2
#delta_c1= (output[:,5]-y_hat[:,5])**2
#delta_c2= (output[:,6]-y_hat[:,6])**2

#cost = T.sum(x1 * delta_x+x1 * delta_y+x1 * delta_w+x1 * delta_h+x1 * delta_c1+x1 * delta_c2)

cost1 = T.sum(x2*delta_xywh + x3*delta_class) #wrong (16,4)+(16,2)
cost2 = T.sum(x2*delta_xywh)
cost3 = T.sum(x3*delta_class)
cost4 = cost2 + cost3

dw = T.grad(cost,w)

f = theano.function(inputs=[x,y_hat],
                    outputs=[output,cost],
                    updates=MyUpdate([w],[dw]))

data=np.ones((1,28*28))
label=np.ones((16,7))
print f(data,label)[0].shape
print f(data,label)


'''
x = T.matrix('x')
y_hat = T.matrix('y_hat')

cost = (x[:,4]-y_hat[:,4]) * ( (x[:,0:4]-y_hat[:,0:4]) + (x[:,5:-1]-y_hat[:,5:-1]))

cost1 = x-y_hat
cost2 = (x[:,4]-y_hat[:,4])
cost3 = (x[:,4]-y_hat[:,4]) * (x[:,4]-y_hat[:,4])
cost4 = (x[:,5]-y_hat[:,5])
cost5 = cost3 * cost4


g=theano.function(inputs=[x,y_hat],
                  outputs=[cost5])

a=np.ones((4*4,5*1+2))
b=np.zeros((4*4,5*1+2))
c=np.ones((4*4,5*1+2))/2
# --
x1=np.ones((4*4,4))
x2=np.asarray([.1,.2,.3,.4,.5,.6,.7,.8,.9,0,.1,.2,.3,.4,.5,.6]).reshape(16,1)
x3=np.zeros((4*4,2))
d=np.append(x1,x2,axis=1)
d=np.append(d,x3,axis=1)
print d

print g(a,d)[0].shape
print g(a,d)
'''


#g=theano.function(inputs=[x,y_hat],
#                    outputs=[dnn.output,cost],
#                    updates=MyUpdate(params,gparams,learning_rate,weight_decay))

#output=np.zeors((4*4,5*1+2))
#label x,y,w,h,c0,class to x,y,w,h,c15,class
# cost function

