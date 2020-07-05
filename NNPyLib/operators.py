from NNPyLib.lib import Function
from NNPyLib.lib import Tensor
import numpy as np

class Linear(Function):
    def __init__(self,in_nodes,out_nodes):
        self.weights = Tensor((in_nodes,out_nodes))
        self.bias    = Tensor((1,out_nodes))
        self.type = 'linear'

    # Y = X*W+b
    def forward(self,x):
        output = np.dot(x,self.weights.data)+self.bias.data
        self.input = x 
        return output

    def backward(self,d_y):
        self.weights.grad += np.dot(self.input.T,d_y)
        self.bias.grad    += np.sum(d_y,axis=0,keepdims=True)
        grad_input         = np.dot(d_y,self.weights.data.T)
        return grad_input

    def getParams(self):
        return [self.weights,self.bias]

class  ReLU(Function):
    def __init__(self,inplace=True):
        self.type    = 'activation'
        self.inplace = inplace
    
    def forward(self,x):
        if self.inplace:
            x[x<0] = 0.
            self.activated = x
        else:
            self.activated = x*(x>0)
        
        return self.activated

    def backward(self,d_y):
        return d_y*(self.activated>0)