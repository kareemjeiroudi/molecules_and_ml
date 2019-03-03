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
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
import pprint
## permute hyperparameters (equivelent of nested for loops)
from itertools import product

from sklearn.model_selection import GridSearchCV # not used anu longer
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

from datetime import datetime # to time the Grid Search

## Load Datset
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)


## Model Definition
def build_model(optimizer, learning_rate, activation, dropout_rate,
                initilizer, num_unit, num_layers=1):
    """
    Wrapper function that takes hyperparametrs as input and 
    returns a keras sequential model
    @Params:
    optimizer:      categorical value e.g. Adam, SGD.
    learning_rate : the learning at which the model updates weights.
    activation:     choice of activation function for hidden layers.
    dropout_rate:   dropout rate at which some neurons of previous layers 
                    are silenced.Used only for hidden layers.
    initializer:    initialization method of input and hidden layers.
    num_unit:       number of neurons in hidden layers. Decreases by half as much
                    for every other layer.
    num_layers:     number of layers of which the hidden layers of the network constitute.
    """
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(num_unit, input_dim=train_samples.shape[1], kernel_initializer=initilizer,
                    activation=activation))
    model.add(Dropout(dropout_rate))
    ## to create an arbitrary number of hidden layers
    for n in range(num_layers):
      d = 2
      nu = int(num_unit/d)
      model.add(Dense(nu, kernel_initializer=initilizer,
                    activation=activation))
      model.add(Dropout(dropout_rate))
      d += 2
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer=optimizer(lr=learning_rate),
                  metrics=['accuracy'])
    return model

## fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## Define the parameters
optimizer = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax]
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
activation = ['relu', 'softsign', 'tanh', 'sigmoid', 'selu']
dropout_rate = [0.0, 0.1, 0.5, 0.7]
initilizer = ['lecun_uniform', 'normal', 'uniform', 'he_uniform']
num_unit = [5, 10, 100, 256, 512, 2048]
num_layers = [2, 5]
epochs = [1, 20, 50, 100]
batch_size = [20, 50, 100, 200, 500, 1000]
momentum = [0.0, 0.2, 0.6,  0.9][:1]

f = open("OUTPUT_02.txt", "w")

## estimate the time needed for the grid search
def estimateTime():
  hyperparameters = [optimizer, learning_rate, activation, dropout_rate,
  initilizer, num_unit, num_layers, epochs, batch_size , momentum]
  product = 1
  for hyperparameter in hyperparameters:
    product = product * len(hyperparameter)
  print("Number of iterations: {}".format(product))
  f.write("Number of iterations: {}\n".format(product))
  seconds = product * 6
  h = seconds//(60*60)
  m = (seconds-h*60*60)//60
  s = seconds-(h*60*60)-(m*60)
  print("Estimated time: {}h:{}m:{}s".format(h, m, s))
  f.write("Estimated time: {}h:{}m:{}s\n".format(h, m, s))

estimateTime()


def gridSearch(hyperparameters_combinations):
  '''
  takes permuations of hyperparameters from the class iterttols.product,
  builds a new model in accord with these parameters,
  returns best auc score and best parameters
  Also prints out all parameters it tests
  @Params:
  hyperparameters_combinations:   a product from class itertools representing all 
                                  possible permutations of hyperparameters.
  '''
  best_score = 0
  params = {}
  best_params = {}
  model_nr = 0
  for comb in hyperparameters_combinations:
    model_nr +=1
    model = build_model(comb[0], comb[1], comb[2], comb[3],
                comb[4], comb[5], comb[6])
    for epoch in range(comb[7]):
      model.fit(train_samples, train_labels, batch_size=comb[8], epochs=1)
      predictions = model.predict(valid_samples)
      auc = roc_auc_score(valid_labels, predictions)

    params['optimizer'] = comb[0]
    params['learning_rate'] = comb[1]
    params['activation'] = comb[2]
    params['dropout_rate'] = comb[3]
    params['initilizer'] = comb[4]
    params['num_unit'] = comb[5]
    params['num_layers'] = comb[6]
    params['epochs'] = comb[7]
    params['batch_size'] = comb[8]
    params['momentum'] = comb[9]
    f.write("Model number {}:\n".format(model_nr))
    f.write(str(params) +'\n')
    f.write('auc score: {}'.format(auc)+'\n\n')

    if best_score < auc:
      best_score = auc
      best_params = params
  return best_score, best_params

hyperparameters_combinations = product(optimizer, learning_rate, activation, dropout_rate,
initilizer, num_unit, num_layers, epochs, batch_size , momentum)

start=datetime.now()
best_auc, best_params = gridSearch(hyperparameters_combinations)
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

