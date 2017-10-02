
# coding: utf-8

# In[ ]:


#!/usr/bin/python2.7

import numpy as np
import pandas as pd
import time as time
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

from check_files import path_check
from predictor import Predictor

def train_nn():

    default_cleaned_dir = './data/cleaned/'
    cleaned_directory = raw_input("Enter cleaned data directory ({0}):".
                                  format(default_cleaned_dir)) or default_cleaned_dir
    # cleaned_directory = default_cleaned_dir

    wf_num = 7
    prefix = 'train'
    error = np.zeros(wf_num)
    trainRatio = 0.8
    error_sum = 0
    
    start = time.time()
    
    for num in range(1, wf_num+1):
        wf = np.loadtxt(cleaned_directory+'{0}_wf{1}.csv'.format(prefix, num), delimiter=',')

        data = wf
        np.random.shuffle(data)

        dataTrain = data[:int(trainRatio*data.shape[0]),:]
        dataTrainX = dataTrain[:,:-1]
        dataTrainX = preprocessing.scale(dataTrainX)
        dataTrainY = dataTrain[:,-1]

        dataTest = data [int(trainRatio*data.shape[0]):,:]
        dataTestX = dataTest[:,:-1]
        dataTestX = preprocessing.scale(dataTestX)
        dataTestY = dataTest[:,-1]

        assert dataTrain.shape[0]+dataTest.shape[0] == data.shape[0]

        hidden_layer_sizes = (10,5)
        learning_rate_init = 0.001
        # activationRange = ['logistic', 'tanh', 'relu']
        activation = 'logistic'

        MLPR = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes,
                            learning_rate_init = learning_rate_init, activation = activation)

        MLPR.fit(dataTrainX, dataTrainY)
        errorTest = dataTestY - MLPR.predict(dataTestX)
        error_sum += np.square(errorTest).sum()
        MSE = np.sqrt(np.square(errorTest).sum()/errorTest.shape[0])
        print 'MSE for wf{0}: {1}'.format(num, MSE)

    MSE_all = np.sqrt(error_sum/errorTest.shape[0]/wf_num)
    print 'MSE for all: {0}'.format(MSE_all)

    end = time.time()
    elapsed = end -start
    minutes, seconds = divmod(elapsed, 60)
    
    print 'Done. Time = {:0>1}m {:05.2f}s.'.format(int(minutes), seconds)
    
def train_nn2():
    """
    Train the nerual network with cleaned data.
    Data is first normalized, and then trained with 
    predictor class from Hongshan.
    
    """

    default_cleaned_dir = './data/cleaned/'
    cleaned_directory = raw_input("Enter cleaned data directory ({0}):".
                                  format(default_cleaned_dir)) or default_cleaned_dir
    # cleaned_directory = default_cleaned_dir

    wf_num = 7
    prefix = 'train'
    error = np.zeros(wf_num)
    trainRatio = 0.8
    
    start = time.time()
    
    for num in range(1, wf_num+1):
        wf = np.loadtxt(cleaned_directory+'{0}_wf{1}.csv'.format(prefix, num), delimiter=',')
            
        dataX = wf[:,:-1]
        dataX = preprocessing.scale(dataX)
        dataY = wf[:,-1]
        dataY = dataY[...,None]
        data = np.append(dataX,dataY,axis=1)
        np.random.shuffle(data)
        
        print "For wf{0}".format(num)
        wfpredictor = Predictor(shape = [17, 10, 5, 1], 
                                path = cleaned_directory,
                                name = '{0}_wf{1}.csv'.format(prefix, num),
                                train_num = int(trainRatio*data.shape[0]))
        wfpredictor.train(num_epoch=2, batch_size=20)
        error[num -1] = wfpredictor.current_error
        
    size_test = data.shape[0] - int(trainRatio*data.shape[0])
    error_sum = (np.square(error)*size_test).sum()
    MSE_all = np.sqrt(error_sum/size_test/wf_num)
    print 'MSE for all: {0}'.format(MSE_all)
    
    end = time.time()
    elapsed = end -start
    minutes, seconds = divmod(elapsed, 60)
    
    print 'Done. Time = {:0>1}m {:05.2f}s.'.format(int(minutes), seconds)
if __name__ == '__main__':
    
    train_nn()
    train_nn2()

