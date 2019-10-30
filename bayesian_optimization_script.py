from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam, RMSprop, Adagrad

from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

import time

import pickle

from bayes_opt import BayesianOptimization


param_reference = {'init': ['lecun_uniform', 'he_uniform', 'uniform', 'glorot_uniform', 'he_normal', 'normal', 'glorot_normal'],
             'activation': ['relu', 'sigmoid', 'selu'],
              'optimizer': ['SGD', 'Adam', 'RMSprop']
                  }

search_space = {'units': (5, 2048), # discrete
                'activation': (0, 2), # categorical
                'optimizer': (0, 2), # categorical
                'lr': (.00001, .2), # continuous
                'n_layers': (2, 15), # discrete
                'epochs': (5, 30), # discrete
                'batch_size': (1, 200), # discrete
                'momentum': (0.001, 0.5), # continuous
                'init': (0, 6), # categorical
                'dropout_rate': (0.0001, 0.9) # continuous
               }


with open('dumped_objects/bits.pckl', 'rb') as f:
    X_train, X_valid, y_train, y_valid = pickle.load(f)


input_dim = X_valid.shape[1]
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


_start_time = time.time()
_i = 0
def tic():
    global _start_time 
    global _i
    _start_time = time.time()
    _i = 1

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Iter {} Time passed: {}hour:{}min:{}sec'.format(_i, t_hour,t_min,t_sec))
    _i+=1


tic()
optimization = BayesianOptimization(f=train_evaluate_and_auc, pbounds=search_space, random_state=0, verbose=0)
optimization.maximize(init_points=20, n_iter=40)
print("Bayesian Optimization took")
tac()
print(optimization.max)


def convert_to_interpretable_dict(optimization_max):
    """ Returns an interpretable dictionary given the a single dictionary from the Bayesian Optimization object above
    """
    interpretable_dict = {}
    interpretable_dict['AUC'] = optimization_max['target']
    interpretable_dict['activation'] = param_reference['activation'][int(round(optimization_max['params']['activation']))] # categorical
    interpretable_dict['batch_size'] = int(round(optimization_max['params']['batch_size']))
    interpretable_dict['dropout_rate'] = optimization_max['params']['dropout_rate']
    interpretable_dict['epochs'] = int(round(optimization_max['params']['epochs']))
    interpretable_dict['init'] = param_reference['init'][int(round(optimization_max['params']['init']))] # categorical
    interpretable_dict['lr'] = optimization_max['params']['lr']
    interpretable_dict['momentum'] = optimization_max['params']['momentum']
    interpretable_dict['n_layers'] = int(round(optimization_max['params']['n_layers']))
    interpretable_dict['optimizer'] = param_reference['optimizer'][int(round(optimization_max['params']['optimizer']))] # categorical
    interpretable_dict['units'] = int(round(optimization_max['params']['units']))
    return interpretable_dict


## save the hyperparameter optimizaiton summary
interpretable_dicts = {}
for param, i in zip(optimization.res, range(len(optimization.res))):    
    interpretable_dicts[i+1] = convert_to_interpretable_dict(param)
param_summary = pd.DataFrame.from_dict(interpretable_dict, orient='index')
param_summary.to_csv('analysis_best_model/param_summary.csv')