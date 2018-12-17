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
                initilizer, num_unit):
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(num_unit, input_dim=train_samples.shape[1], kernel_initializer=initilizer,
                    activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_unit, kernel_initializer=initilizer,
                    activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer(lr=learning_rate),
                  metrics=['accuracy'])
    return model


## Define the parameters
batch_size = [20, 50, 100]
epochs = [1, 20, 50]
initilizer = ['lecun_uniform', 'normal', 'he_normal', 'he_uniform']
learning_rate = [0.1, 0.001, 0.02]
dropout_rate = [0.3, 0.2, 0.8]
num_unit = [10, 5]
activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
optimizer = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]

## Model wrapper and GridSearchCV
parameters = dict(batch_size=batch_size,
                  epochs=epochs,
                  dropout_rate=dropout_rate,
                  num_unit=num_unit,
                  initilizer=initilizer,
                  learning_rate=learning_rate,
                  activation=activation,
                  optimizer=optimizer)

model = KerasClassifier(build_fn=build_model, verbose=0)
start=datetime.now()
models = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1)

## Train the models
best_model = models.fit(train_samples, train_labels)
timing = datetime.now()-start
print(timing)
print('Best model :')
pp.pprint(best_model.best_params_)
pp.pprint(best_model.best_score_)

f = open("OUTPUT_01.txt", "a")
f.write('Best model:\n')
f.write(str(best_model.best_params_)+"\n")
f.write(str(best_model.best_score_)+"\n")
f.write(str(timing)+"\n")

f.close()

