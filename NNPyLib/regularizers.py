import numpy as np
from NNPyLib.lib import Function

class Dropout(Function):

    def __init__(self,prob=0.5):
        self.type = 'regularization'
        self.prob = prob
        self.params = []

    def forward(self,X):
        self.mask = np.random.binomial(1,self.prob,size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)
    
    def backward(self,dout):
        dX = dout * self.mask
        return dX