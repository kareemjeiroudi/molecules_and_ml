# to suppress the FutureWarning: conversion of the second argument of issubdtype 
# from 'float' to 'np.floatin' is deprecated
import os
import numpy as np
## importing all libraries that we'd need
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
## for Grid Search
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores


## Load the data
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)


def create_model(learn_rate=0.01, momentum=0, init_mode='uniform', activation='relu',
                dropout_rate=0.0, weight_constraint=0, neurons=1):
    '''Function to create model, required for KerasClassifier'''
    ## create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_samples.shape[1], kernel_initializer=init_mode, activation=activation,
                       kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(int(round(neurons/4)), activation='selu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid')) # output layer
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    ## Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


## fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 
             'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
              'sigmoid', 'hard_sigmoid']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [5, 10, 20, 50, 100, 200, 256, 512, 2048]

param_grid = dict(batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, momentum=momentum,
                 init_mode=init_mode, activation=activation, dropout_rate=dropout_rate,
                  weight_constraint=weight_constraint, neurons=neurons)

## create model
model = KerasClassifier(build_fn=create_model, verbose=0)
## time the iterations
from datetime import datetime
start=datetime.now()
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1) # -1 to run in parallel
grid_result = grid.fit(valid_samples, valid_labels)
print(datetime.now()-start)

## summarize results
f = open("OUTPUT.txt", "a")
line = "Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_)
print(line)
f.write(line + "\n")
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
  line = "{0} ({1}) with: {2}".format(mean, stdev, param) 
  print(line)
  f.write(line + "\n")
f.close()


