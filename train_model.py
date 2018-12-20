# to suppress the FutureWarning: conversion of the second argument of issubdtype 
# from 'float' to 'np.floatin' is deprecated
import os

import numpy as np
# importing all libraries that we'd need
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.metrics import categorical_crossentropy

from sklearn import cross_validation
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

## for Grid Search
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm


## Load data
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)

## Build Model
backend.clear_session()
model = Sequential()
initializer = 'lecun_uniform'
model.add(Dense(256, input_dim=train_samples.shape[1], activation='relu',
               kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(units=int(256/8), activation='hard_sigmoid',
                kernel_initializer=initializer))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))


# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=["accuracy"])
model.compile(loss='binary_crossentropy',
              optimizer=Adadelta(lr=0.1), metrics=['accuracy'])

model.summary()

for epoch in range(50):
    model.fit(train_samples, train_labels, batch_size=20, epochs=1)
    predictions = model.predict(valid_samples)
    auc = roc_auc_score(valid_labels, predictions)
    print(auc)