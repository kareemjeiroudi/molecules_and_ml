from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

from sklearn.metrics import roc_auc_score
import csv

import numpy as np
import pandas as pd

import time

import pickle

from bayes_opt import BayesianOptimization

## load train and validation matrix
with open('dumped_objects/bits.pckl', 'rb') as f:
    X_train, y_train, X_valid, y_valid, _, _ = pickle.load(f)

print("Loaded train and validation matrices")
## Hyper parameter search space
param_reference = {'init': ['lecun_uniform', 'uniform', 'normal'],
             'activation': ['relu', 'sigmoid', 'selu'],
              'optimizer': ['SGD', 'Adam', 'RMSprop']
                  }

search_space = {'units': (5, 1048), # discrete
                'activation': (0, 2), # categorical
                'optimizer': (0, 1), # categorical
                'lr': (.00001, .2), # continuous
                'n_layers': (2, 15), # discrete
                'epochs': (5, 30), # discrete
                'batch_size': (1, 200), # discrete
                'momentum': (0.001, 0.5), # continuous
                'init': (0, 2), # categorical
                'dropout_rate': (0.0001, 0.9) # continuous
               }
print("Defined hyper parameter settings search space")


_start_time = time.time()
def tic():
    global _start_time 
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

input_dim = X_valid.shape[1]
## define optimizatoin traget function
def train_evaluate_and_auc(units, activation, optimizer, lr, n_layers, epochs, batch_size, momentum, init, dropout_rate):
    # handle categorical
    activation = param_reference['activation'][int(round(activation))]
    optimizer = param_reference['optimizer'][int(round(optimizer))]
    init = param_reference['init'][int(round(init))]
    # handle discrete
    units = int(round(units))
    n_layers = int(round(n_layers))
    epochs = int(round(epochs))
    batch_size = int(round(batch_size))
    tac()
    
    # build classifier with corresponding hyperparameters
    k.clear_session()
    classifier = Sequential()
    classifier.add(
        Dense(units=units, input_dim=input_dim, activation=activation, kernel_initializer=init) # input layer
    )
    # Create an arbitrary number of Hidden Layers
    for n in range(n_layers):
        classifier.add(Dense(units=units, activation=activation))
        classifier.add(Dropout(dropout_rate))
    classifier.add(Dense(units=1, activation='sigmoid'))     # Output Layer
    optimizer_eval = eval(optimizer+'(learning_rate={}, momentum={})'.format(lr, momentum)) if optimizer=='SGD' else eval(optimizer+'(learning_rate={})'.format(lr))
    classifier.compile(loss='binary_crossentropy', optimizer=optimizer_eval, metrics=['accuracy'])
    classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    return roc_auc_score(y_valid, classifier.predict(X_valid))
print("Declared optimization target function")



## Create an optimization object
tic()
optimization = BayesianOptimization(f=train_evaluate_and_auc, pbounds=search_space, random_state=0, verbose=0)
print("Started optimization")
optimization.maximize(init_points=10, n_iter=20)
print("Bayesian Optimization took")
tac()
print(optimization.max)

## Dumb the optimization object
with open('dumped_objects/optimization.pckl', 'wb') as f:
    pickle.dump(optimization, f)