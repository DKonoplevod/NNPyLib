import numpy as np
import NNPyLib as lib

if __name__=="__main__":
    batch_size        = 20
    num_epochs        = 100
    samples_per_class = 100
    num_classes       = 3
    hidden_units      = 100
    data,target       = lib.utils.genSpiralData(samples_per_class,num_classes)
    #lib.utils.plot2DData(data,target)
    model             = lib.utils.Model()
    model.add(lib.operators.Linear(2,hidden_units))
    model.add(lib.operators.ReLU())
    model.add(lib.operators.Linear(hidden_units,num_classes))
    optim   = lib.optimizers.SGD(model.parameters,lr=1.0,weight_decay=0.001,momentum=.9)
    loss_fn = lib.loss_functions.SoftmaxWithLoss()
    model.fit(data,target,batch_size,num_epochs,optim,loss_fn)
    predicted_labels = np.argmax(model.predict(data),axis=1)
    accuracy         = np.sum(predicted_labels==target)/len(target)
    print("Model Accuracy = {}".format(accuracy))
    lib.utils.plot2DDataWithDecisionBoundary(data,target,model)