# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:44:12 2019

@author: Nors
"""


"""
Markov Chain Properties:
    1. At any moment, the before and next state are independent
    2. The current step only depends on previous step(s)
This is a SGLD approach, but not "stochastic" at the moment, and without MH
"""

# (point + noise) -> sgd several times
from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np
import csv
import matplotlib.pyplot as plt
from keras import optimizers



def FlattenList(data):
    final = []
    for row in data:
        final = np.concatenate((final, np.reshape(row,(1,row.size))), axis=None)
    return final



def GenerateDataset(x,y,TraceBackCounter):

    X,Y = [],[]
    IterNum = len(x)-TraceBackCounter
    for i in range(IterNum):
        Train = []
        for j in range(i,i+TraceBackCounter):
            Train.append([x[j],y[j]])
        
        X.append(Train)
        Y.append([x[i+TraceBackCounter],y[i+TraceBackCounter]])
    """
    LSTM network expects the input data (X) to be provided with a specific
    array structure in the form of: [samples, time steps, features].
    """
    #return X,Y
    return np.array(X).reshape(IterNum,TraceBackCounter,2),np.array(Y)

def GaussianNoisy(sequence,mu,sig):
    return np.add(sequence,np.random.normal(mu,sig,len(sequence)))

def Concat3D(DataDict):
    X,Y = [],[]
    for name in DataDict:
        if(len(X)==0):
            X = DataDict[name][0]
        else:
            X = np.concatenate((X,DataDict[name][0]))
        if(len(Y)==0):
            Y = DataDict[name][1]
        else:
            Y = np.concatenate((Y,DataDict[name][1]))
        
    return X,Y
    

def GetMSE(Y_true,Y_pred):
    MSE = np.square(np.subtract(Y_true,Y_pred)).mean()
    return MSE

def Decompose(data):
    x,y = [],[]
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
        
    return x,y


if __name__ == '__main__':
    TargetFile = 'counter_clockwise.csv'
    x,y = [],[]
    with open(TargetFile, newline='') as file:
        reader = csv.reader(file)
        for item in reader:
            x.append(float(item[0]))
            y.append(float(item[1]))
    
    
    TraceBackCounter = 5
    NeuronNum = 2
    mu = 0
    sig = 0.05
    TestSig = 0.5
    
    TrainingNum = 5
    TestNum = 1
    
    DataDict = {}
    for i in range(TrainingNum):
        DataDict[str(i)] = GenerateDataset(GaussianNoisy(x,mu,0),GaussianNoisy(y,mu,0),TraceBackCounter) 
    RealSet = Concat3D(DataDict)
    
    DataDict = {}
    for i in range(TrainingNum):
        DataDict[str(i)] = GenerateDataset(GaussianNoisy(x,mu,sig),GaussianNoisy(y,mu,sig),TraceBackCounter) 
    TrainingSet = Concat3D(DataDict)
    
    DataDict = {}
    for i in range(TestNum):
        DataDict[str(i)] = GenerateDataset(GaussianNoisy(x,0,TestSig),GaussianNoisy(y,0,TestSig),TraceBackCounter)
    TestSet = Concat3D(DataDict)
        
    
    BatchSize = len(x)
    model = Sequential()
    #model.add(SimpleRNN(NeuronNum, input_shape = (TraceBackCounter,2), return_sequences = True))
    model.add(LSTM(NeuronNum, input_shape = (TraceBackCounter,2)))
    model.add(Dense(2,activation='tanh'))
    
    
    LearningRate = 0.1
    TrainX = TrainingSet[0]
    TrainY = TrainingSet[1]
    op = optimizers.Adam(lr=LearningRate)
    
    model.compile(loss = 'mean_squared_error', optimizer=op)
    model.fit(TrainingSet[0], TrainingSet[1], epochs = 400, batch_size = BatchSize, verbose = 2)
    
    Prediction = model.predict(TestSet[0])
    PredictionX,PredictionY = Decompose(Prediction)
    GroundTruth = TestSet[1]
    GroundTruthX,GroundTruthY = Decompose(GroundTruth)
    plt.figure()
    plt.plot(PredictionX,PredictionY, label = 'Prediction')
    plt.plot(GroundTruthX,GroundTruthY, label = 'Ground Truth')
    plt.legend(loc='upper right')
    print("Fitting Error = %f" % model.evaluate(TestSet[0],TestSet[1]))
    
    
    TrainingTime = 50000
    LFStep = 5
    ParamDict = {}
    counter = 0     
    RejectionTime = 0       
    for i in range(TrainingTime):
        if i % 100 == 0:
            print("%d of %d" % (i,TrainingTime))
            
        CurrentWeights = model.get_weights()
        ParamDict[str(counter)] = {}
        ParamDict[str(counter)]["Params"] = CurrentWeights
        counter = counter + 1
        CurrentLoss = model.evaluate(TrainX,TrainY)
        
        # New try
        ProposedWeights = CurrentWeights
        for c in range(LFStep):
            noise=[np.random.randn(*w.shape)*LearningRate for w in ProposedWeights] # Generate noise
            ProposedWeights = np.add(ProposedWeights,noise) # Add noise to proposed weights
            model.set_weights(ProposedWeights) # Update the proposed weight with noise to the model
            model.fit(TrainX, TrainY, epochs=1, batch_size=BatchSize, verbose=2) # Get new proposed weights according to gd
            ProposedWeights = model.get_weights() # Update new weights to proposed weights


        ProposedLoss = model.evaluate(TrainX,TrainY)
        dH = np.exp(((CurrentLoss) - (ProposedLoss))*200)
        
        alpha = np.min([1,dH])
        u = np.random.rand()
        if u >= alpha:
            # reject
            model.set_weights(CurrentWeights)
            RejectionTime += 1

    print("Rejection rate = %f" % (RejectionTime / TrainingTime))


    NewModel = Sequential()
    NewModel.add(LSTM(NeuronNum, input_shape = (TraceBackCounter,2)))
    NewModel.add(Dense(2,activation='tanh'))
    
    TestX = TestSet[0]
    TestY = TestSet[1]
    GroundTruth = TestY
    GroundTruthX,GroundTruthY = Decompose(GroundTruth)
    NewX,NewY = [],[]
    NewX.append(GroundTruthX)
    NewY.append(GroundTruthY)
    
    for element in range(len(ParamDict)):
        NewModel.set_weights(ParamDict[str(element)]["Params"])
        PredictionX,PredictionY = Decompose(NewModel.predict(TestX))
        NewX.append(PredictionX)
        NewY.append(PredictionY)
    
    
    
    
    # Write out x and y separately
    with open("PredictionOutputX0.25.csv", 'w',newline='') as file:
        wr = csv.writer(file, dialect='excel')
        for row in NewX:
            wr.writerow(row)
    file.close()
    
    
    with open("PredictionOutputY0.25.csv", 'w',newline='') as file:
        wr = csv.writer(file, dialect='excel')
        for row in NewY:
            wr.writerow(row)
    file.close()
