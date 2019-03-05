#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:20:42 2019

@author: kareem
@date: 05.03.2019
@Title: Reconstructing best scoring Model
"""

## Importing Learning Data (data is already normalized and scaled so no need to make any extra normalization)
import numpy as np
X_train = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
y_train = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
X_valid = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
y_valid = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)
                          
## See about converting a string into a dictionary
dictionary = "{'optimizer': <class 'keras.optimizers.SGD'>, 'learning_rate': 0.1, 'activation': 'relu', 'dropout_rate': 0.1, 'initilizer': 'he_uniform', 'num_unit': 100, 'num_layers': 2, 'epochs': 50, 'batch_size': 500, 'momentum': 0.0}"
### Constructing the model ###
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.metrics import binary_accuracy
k.backend.clear_session()
classifier = Sequential()
## Input layer
classifier.add(Dense(units=100, input_dim=len(X_train[0]), activation='relu', kernel_initializer='he_uniform'))
## Hidden layer
classifier.add(Dense(units=50, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=25, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(rate=0.1))
## Ouptut layer
classifier.add(Dense(units=1, activation='sigmoid'))
## Compilation
classifier.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=[binary_accuracy])
## Train
classifier.fit(X_train, y_train, batch_size=500, epochs=50)
## Predict
y_prob = classifier.predict(X_valid)
y_pred= classifier.predict_classes(X_valid)
## Evaluating AUC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
auc = roc_auc_score(y_valid, y_prob)

## Plotting ROC
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='brown',
             lw=2, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [1, 0], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Best_model_ROC')
    plt.show()
plot_roc_curve(fpr, tpr, auc)

## Choosing a custom threshold
y_label = [0 if y_pred[i]<=0.661766 else 1 for i in range(len(y_pred))]
y_label = np.array(y_label)

## Construct and plot confusion matrix
cm = confusion_matrix(y_valid, y_pred)
from itertools import product
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

np.set_printoptions(precision=2)
        
## Plot Unnormalized confusion matrix
class_names=['non-mutagen', 'mutagen']
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
## Plot normalized confusion matrix
#plot_confusion_matrix(cm, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
plt.show()


## Extracting dictionaries to make a full-record analysis
import re
import ast
indices = []
all_hyperparameters = []
scores = []
f = open('OUTPUT_02.txt', 'r')
for line in f.readlines():
    ## Store model number
    if line.startswith('Model'):
        match = int(re.search(re.compile(" \d+:"), line).group(0)[1:-1])
        indices.append(match)
    if line.startswith('auc'):
        match = float(re.search(re.compile(" 0\.\d+"), line).group(0)[1:])
        scores.append(match)
    ## Store parameters
    if line.startswith('{'):
        match = re.search(re.compile("<.*>"), line).group(0)
        optimizer_name =  "'" + re.search(re.compile("\.\w*'>"), match).group(0)[1:-1]
        line = re.sub(match, optimizer_name, line)        
        all_hyperparameters.append(ast.literal_eval(line))
f.close()

for i in range(all_hyperparameters):
    all_hyperparameters[i]['index'] = index
    all_hyperparameters[i]['auc'] = score
    

## Writing the dictionaries into a csv file
import csv
with open('all_hyperparameters.csv', 'w', newline='') as csvfile:
    fieldnames = all_hyperparameters[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(all_hyperparameters)