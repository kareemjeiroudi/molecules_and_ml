#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:11:57 2019

@author: kareem
@Date: 03.03.2019
@Title: Grid Search for Molecules mutagenity
"""
## Imports
from __future__ import print_function

#from keras.datasets import mnist


from sklearn.model_selection import GridSearchCV # not used anu longer
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

from datetime import datetime # to time the Grid Search


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
from sklearn.metrics import roc_auc_score

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
                  metrics=['accuracy', roc_auc_score])
    
    return model


## Define the Search Space (takes a week)
units = [100, 256, 512, 1024, 2048]
activation = ['relu', 'sigmoid', 'selu']
optimizer = [SGD, Adam]
learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
num_layers = [2, 3, 5, 9]
dropout_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
#initilizer = ['lecun_uniform', 'normal', 'uniform', 'he_uniform']
epochs = [10, 20, 50, 100]
batch_size = [20, 50, 100, 200]
#momentum = [0.0, 0.2, 0.6,  0.9][:1]

hyperparameters = [units,
                   activation,
                   optimizer,
                   learning_rate,
                   num_layers,
                   dropout_rate,
                   epochs,
                   batch_size]

## Define (test) Search Space (takes a whole day)
units = [512, 1024, 2048]
activation = ['relu', 'sigmoid', 'selu']
optimizer = [SGD, Adam]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.5]
num_layers = [2, 5, 9]
dropout_rate = [0.001, 0.01, 0.1, 0.2, 0.5]
#initilizer = ['lecun_uniform', 'normal', 'uniform', 'he_uniform']
epochs = [10, 20, 50, 100]
batch_size = [50, 100, 200]
#momentum = [0.0, 0.2, 0.6,  0.9][:1]

hyperparameters = [units,
                   activation,
                   optimizer,
                   learning_rate,
                   num_layers,
                   dropout_rate,
                   epochs,
                   batch_size]

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
    message = "Estimated time: {}h:{}m:{}s".format(h, m, s)
    print(message)
    f.write(message)

estimateTime()


## Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def gridSearch(hyperparameters):
  '''
  Wrapper function that takes permuations of hyperparameters from the class itertools.product,
  builds a new model in accord with these parameters. Analogous to the GridSerach_CV form 
  sklearn. Lasty, it returns best Roc AUC score and best parameters
  Also prints out all parameters it tests (the progress of the Grid Search).
  
  Parameters:
  -------
  hyperparameters:
      (Iterable) product from class itertools representing all possible permutations of hyperparameters.
  '''
  ## Values to be reported
  best_score = 0
  params = {}
  best_params = {}
  model_nr = 0
  
  for permute in hyperparameters:
    model_nr +=1
    params['units'] = permute[0]
    params['activation'] = permute[1]
    params['num_layers'] = permute[2]
#    params['dropout_rate'] = permute[3]
    params['optimizer'] = permute[3]
    params['learning_rate'] = permute[4]
    params['epochs'] = permute[5]
    params['batch_size'] = permute[6]
    
    model = build_classifier(permute[0], permute[1], permute[2], permute[3],
                permute[4], permute[5], permute[6])
    for epoch in range(permute[7]):
      model.fit(train_samples, train_labels, batch_size=permute[8], epochs=1)
      predictions = model.predict(valid_samples)
      auc = roc_auc_score(valid_labels, predictions)

   
    f.write("Model number {}:\n".format(model_nr))
    f.write(str(params) +'\n')
    f.write('auc score: {}'.format(auc)+'\n\n')

    if best_score < auc:
      best_score = auc
      best_params = params
  return best_score, best_params

## Permute hyperparameters (equivelent of nested for loops)
from itertools import product
hyperparameters = product(units,
                   activation,
                   num_layers,
#                   dropout_rate,
                   optimizer,
                   learning_rate,
                   epochs,
                   batch_size)
for i in hyperparameters:
    print(i)
start=datetime.now()
best_auc, best_params = gridSearch(hyperparameters)
timing = datetime.now()-start
print("The Search took {}".format(timing))
print('Best model:')
print(best_auc)
print(best_params)

f.write('\n\nBest model:\n')
f.write(str(best_params)+"\n")
f.write(str(best_auc)+"\n")
f.write("The search took: {}".format(timing)+"\n")

f.close()

