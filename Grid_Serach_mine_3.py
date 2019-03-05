#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:11:57 2019

@author: kareem
@Date: 04.03.2019
@Title: Grid Search for Molecules Mutagenity
"""

## Load Datset
import numpy as np
X_train = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
y_train = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
X_test = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
y_test = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)

              
## Model Definition
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
def build_classifier(units, activation, optimizer, learning_rate, num_layers=1, dropout_rate=0.2):
    """
    Wrapped function that takes hyperparametrs as input and 
    returns a desired Keras sequential model. Useful for a Grid Search.
    
    Params:
    -------
    units:
        (int) number of neurons in the hidden layers (i.e. output dimension).
    activation:    
        categorical - string, choice of activation function for hidden layers.
    optimizer:
        categorical - string value e.g. Adam, SGD.
    learning_rate :
        (float) the learning at which the model updates weights.
    num_layers:
        (int) number of hidden layers. By default=1
    dropout_rate:   
        (float) dropout rate at which some neurons of previous layers are silenced. Used only for hidden layers.
    
    Returns:
    --------
        Keras classifier model
    """
    ## clear session to avoid overlaoding the backend engine
    keras.backend.clear_session()
    model = Sequential()
    ## Input layers
    model.add(Dense(units=units, input_dim=X_train.shape[1], activation=activation))
    ## Create an arbitrary number of Hidden Layers
    for n in range(num_layers):
#      d = 2
#      nu = int(units/d)
      model.add(Dense(units=units, activation=activation))
      model.add(Dropout(dropout_rate))
      ## end of for loop
    
    ## Output Layer
    model.add(Dense(units=1, activation='sigmoid'))
    ## Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer(lr=learning_rate),
                  metrics=['accuracy'])
    
    return model


## Define  a 3rd (test) Search Space (take about 1.5 hours)
units = [512, 1024, 2048]
activation = ['relu', 'sigmoid', 'selu']
optimizer = [SGD, Adam]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.5]
num_layers = [2, 5, 9]
#dropout_rate = [0.001, 0.01, 0.1, 0.2, 0.5]
#initilizer = ['lecun_uniform', 'normal', 'uniform', 'he_uniform']
epochs = [10]
batch_size = [50, 100, 200]
#momentum = [0.0, 0.2, 0.6,  0.9][:1]

hyperparameters = [units,
                   activation,
                   optimizer,
                   learning_rate,
                   num_layers,
#                   dropout_rate,
                   epochs,
                   batch_size]

## the folloing function requires that the output file is open
f = open("Grid_Search_Ouput.txt", "w")
def estimateTime():
    """
    Custom function that estimates the time that a grid search will take depending on the
    number of permutations.
    """
    product = 1
    for hyperparameter in hyperparameters:
        print(len(hyperparameter))
        product = product * len(hyperparameter)
    message = "Number of iterations: {}".format(product)
    print(message)
    f.write(message)
    total_seconds = product * 6
    h = str(total_seconds//3600)
    remainder = float('.'+str(total_seconds/3600).split('.')[1])
    m = str(format(60 * remainder, '.8f')).split('.')[0]
    remainder = float(str(format(60 * remainder, '.8f')).split('.')[1])
    s = str(format(remainder * 60, '.8f')).split('.')[0]
    message = "Estimated time: {}h:{}m:{}s\n\n".format(h, m, s)
    print(message)
    f.write(message)

estimateTime()

## Permute hyperparameters (equivelent of nested for loops)
from itertools import product
hyperparameters = product(units,
                   activation,
                   optimizer,
                   learning_rate,
                   num_layers,
#                   dropout_rate,
                   epochs,
                   batch_size)
hyperparameters = {
        'units':units,
        'activation': activation,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'num_layers': num_layers,
        'epochs': epochs,
        'batch_size': batch_size
        }

## Conduct Grid Search
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
## Fix random seed for reproducibility
seed = 7
np.random.seed(seed)
gridSearch = GridSearchCV(estimator=build_classifier, param_grid=hyperparameters, scoring='roc_auc', cv=5, n_jobs=-1)

## Start timing the Grid Search
from datetime import datetime # to time the Grid Search
start=datetime.now()
best_auc, best_params, best_model = gridSearch(hyperparameters)
timing = datetime.now()-start

## Report timing to both terminal and output file
print("Grid Search took {}".format(str(timing)))
f.write("Grid Search took: {}".format(str(timing))+"\n")

f.close()

