import h5py
import numpy as np
train=h5py.File('../input/train_catvnoncat.h5',mode='r')
train_feat=np.array(train['train_set_x'][:])
train_lable=np.array(train['train_set_y'][:])
test=h5py.File('../input/test_catvnoncat.h5',mode='r')
test_feat=np.array(test['test_set_x'][:])
test_lable=np.array(test['test_set_y'][:]) 
classes=np.array(test['list_classes'][:])
train_lable=train_lable.reshape((1,train_lable.shape[0]))
test_lable=test_lable.reshape((1,test_lable.shape[0]))
train_feat=train_feat.reshape(train_feat.shape[0],-1).T/255
test_feat=test_feat.reshape(test_feat.shape[0],-1).T/255
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def initialize_with_zeros(dim):
    w = np.zeros(shape = (dim,1))
    b = 0
    assert(w.shape == (dim, 1)) 
    assert(isinstance(b, float) or isinstance(b, int)) 
    return (w , b)

def propagate(X,Y,W,b):
    n=X.shape[1]
    yh=sigmoid(np.dot(W.T,X)+b)
    cost=-(1/m)*np.sum(Y*np.log(yh)+(1-Y)*np.log(1-yh))
    dw=np.dot(X,(yh-Y).T)/m
    db=np.sum(yh-Y)/m
    cost=np.squeeze(cost)
    grad_dict={
        'dw':dw,
        'db':db
    }
    return cost,grad
 def optimize(X,Y,W,b,num_iterations,learning_rate,print_cost=False):
    costs=[]
    for i in num_iterations:
        cost,grad=propagate(X,Y,W,b)
        dw,db=grad['dw'],grad['db']
        W=W-learning_rate*dw
        b=b-learning_rate*db
        if i%50==0:
            costs.append(cost)
        if print_cost==True and (i%50)==0:
            print('迭代的次数{0},误差值{1}'.format(i,cost))
    params={W:W,b:b}
    grads={dw:dw,db:db}
    return costs,params,grads
def predict(X,W,b):
    m=X.shape[1]
    y_pred=np.zeros((1,m))
    W=W.reshape((X.shape[0],1))
    a=sigmoid(np.dot(W.T,X)+b)
    for i in range(X.shape[1]):
        y_pred[0,i]=1 if a==0.5 else 0
    assert(y_pred.shape==(1,m))
    return y_pred
