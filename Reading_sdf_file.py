# to suppress the FutureWarning: conversion of the second argument of issubdtype 
# from 'float' to 'np.floatin' is deprecated
import os
import numpy as np
## importing all libraries that we'd need
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout
# from talos.layers import hidden_layers
from keras.layers.core import Dense
from keras.activations import relu, elu, softmax
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.metrics import categorical_crossentropy
from keras.metrics import fmeasure
from keras.losses import binary_crossentropy, logcosh
from keras.constraints import maxnorm
## for Grid Search
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
## these are from talos
import talos as ta
import wrangle as wr
from talos.metrics.keras_metrics import fmeasure_acc
from talos import live


## Load the data
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)


# first we have to make sure to input data and params into the function
def create_model(train_samples, train_labels, valid_samples, valid_labels, params):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=train_samples.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    model.add(Dropout(params['dropout']))
    # if we want to also test for number of layers and shapes, that's possible
    hidden_layers(model, params, 1) # 1 for number of hidden layers
    # then we finish again with completely standard Keras way
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer='normal'))
    
    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['acc', fmeasure])
    
    history = model.fit(train_samples, train_labels, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)
    
    # finally we have to make sure that history object and model are returned
    return history, model


## fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# then we can go ahead and set the parameter space
params = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [150],
     'dropout': [0, 0.2, 0.5],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','long_funnel'],
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation':[relu, elu],
     'last_activation': ['sigmoid']
     }

## time the iterations
from datetime import datetime
start=datetime.now()
t = ta.Scan(train_samples, train_labels,
  model=create_model,
  grid_downsample=0.01,
  params=params,
  dataset_name="Morgan fingerprints",
  experiment_no='1')
print(datetime.now()-start)

r = ta.Reporting("OUTPUT_01.csv")
## summarize results
f = open("OUTPUT_01.txt", "a")
# line = "Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_)
# accessing the results data frame
line = h.data.head()
print(line)
f.write(line + "\n")
# accessing epoch entropy values for each round
line = h.peak_epochs_df
print(line)
f.write(line +"\n")
# access the summary details
line = h.details
print(line)
f.write(line +"\n")
# accessing the saved models
line = h.saved_models
print(line)
f.write(line +"\n")
# accessing the saved weights for models
line=saved_weights
print(line)
f.write(line +"\n")

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):
  # line = "{0} ({1}) with: {2}".format(mean, stdev, param) 
  # print(line)
  # f.write(line + "\n")
f.close()



