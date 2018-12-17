# to suppress the FutureWarning: conversion of the second argument of issubdtype 
# from 'float' to 'np.floatin' is deprecated
import os

# importing all libraries that we'd need
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
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