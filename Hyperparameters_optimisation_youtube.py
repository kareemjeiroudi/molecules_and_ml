## imports
from __future__ import print_function
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
from keras.datasets import mnist
import pprint

from itertools import product

from sklearn.model_selection import GridSearchCV

from datetime import datetime # to time the Grid Search

pp = pprint.PrettyPrinter(indent=4)

## Load Datset
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)


## Model Definition
def build_model(optimizer, learning_rate, activation, dropout_rate,
                initilizer, num_unit, num_layers):
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(num_unit, input_dim=train_samples.shape[1], kernel_initializer=initilizer,
                    activation=activation))
    model.add(Dropout(dropout_rate))
    for n in range(num_unit):
    model.add(Dense(num_unit, kernel_initializer=initilizer,
                    activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer(lr=learning_rate),
                  metrics=['accuracy'])
    return model

## fix random seed for reproducibility
seed = 7
np.random.seed(seed)

## Define the parameters
batch_size = [20, 50, 100, 200, 500, 1000]
epochs = [1, 20, 50, 100]
initilizer = ['lecun_uniform', 'normal', 'uniform', 'he_uniform']
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.6,  0.9]
dropout_rate = [0.0, 0.1, 0.5, 0.7]
num_unit = [5, 10, 100, 256, 512, 2048]
activation = ['relu', 'softsign', 'tanh', 'sigmoid', 'linear', 'selu']
optimizer = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax]

## estimate the time needed for the grid search
hyperparameters = [optimizer, learning_rate, activation, dropout_rate,
initilizer, num_unit, batch_size, epochs, momentum]
product = 1
for hyperparameter in hyperparameters:
  product *= len(hyperparameter)
print("Number of iterations: {}".format(product))
seconds = product * 40
h = seconds//(60*60)
m = (seconds-h*60*60)//60
s = seconds-(h*60*60)-(m*60)
print("Estimated time: {}h:{}m:{}s".format(h, m, s))


## Model wrapper and GridSearchCV
# parameters = dict(batch_size=batch_size,
#                   epochs=epochs,
#                   dropout_rate=dropout_rate,
#                   num_unit=num_unit,
#                   initilizer=initilizer,
#                   learning_rate=learning_rate,
#                   activation=activation,
#                   optimizer=optimizer)

## create the model
# model = KerasClassifier(build_fn=build_model, verbose=0)
start=datetime.now()
# models = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1)

def gridSearch(hyperparameters_combinations):
  best_score = 0
  best_params = {}
  for comb in hyperparameters_combinations:
    build_model(comb[0], comb[1], comb[2], comb[3],
                comb[4], comb[5])

hyperparameters_combinations = product(batch_size, epochs, initilizer, learning_rate,
momentum, dropout_rate, num_unit, activation)

gridSearch(hyperparameters_combinations)

## Train the models
best_model = models.fit(train_samples, train_labels)
timing = datetime.now()-start
print(timing)
print('Best model :')
pp.pprint(best_model.best_params_)
pp.pprint(best_model.best_score_)

f = open("OUTPUT_02.txt", "w")
f.write('Best model:\n')
f.write(str(best_model.best_params_)+"\n")
f.write(str(best_model.best_score_)+"\n")
f.write(str(timing)+"\n")

f.close()

