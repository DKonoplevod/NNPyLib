import numpy as np

class Tensor():
    def __init__(self,shape):
        self.data = np.ndarray(shape,np.float32)
        self.grad = np.ndarray(shape,np.float32)

# Operator interface
class  Function(object):
    def forward(self): 
        raise NotImplementedError
    
    def backward(self): 
        raise NotImplementedError
    
    def getParams(self): 
        return []

# Optimizers interfacce
class Optimizer(object):
    def __init__(self,parameters):
        self.parameters = parameters
 
    def step(self): 
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.parameters:
            p.grad = 0.

##############################Library Implementation **END##############################