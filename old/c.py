import numpy as np

def save():
  b=np.zeros((20,20))
  x=np.zeros((5,5))
  x[1,2],x[2,1],x[2,2],x[2,3],x[3,2]=1,1,1,1,1
  label=np.eye(16)

  f=open('train','a')
  ctr=0
  for j in range(4):
    for i in range(4):
      b[5*j:5*(j+1),5*i:5*(i+1)]+=x
      c=np.copy(b).reshape((1,400))
      c=np.concatenate([c,label[ctr].reshape((1,16))],axis=1)
      np.savetxt(f,c,fmt="%d")
      b[5*j:5*(j+1),5*i:5*(i+1)]-=x
      ctr+=1
  f.close()

if __name__=='__main__':
  save()
  k=np.loadtxt('train')
  print k.shape
  for d in range(k.shape[0]):
    t,l=k[d,0:400].reshape(20,20),k[d,400:]
    for i in range(20):
      for j in range(20):
        if t[i,j]:
          print i,j,l.argmax()
    print '\n'
