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
X_test = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
y_test = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)

### Constructing the model ###
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
k.backend.clear_session()
classifier = Sequential()
## Input layer
classifier.add(Dense(units=512, input_dim=len(X_train[0]), activation='relu'))
## Hidden layer
for i in range(5):
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(rate=0.1))
## Ouptut layer
classifier.add(Dense(units=1, activation='sigmoid'))
## Compilation
classifier.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=[binary_accuracy])
## Train
classifier.fit(X_train, y_train, batch_size=100, epochs=10)
## Predict
y_prob = classifier.predict(X_test)
y_pred= classifier.predict_classes(X_test)
## Evaluating AUC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
auc = roc_auc_score(y_test, y_prob)

## Plotting ROC
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='brown',
             lw=2, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Best_model_ROC')
    plt.show()
plot_roc_curve(fpr, tpr, auc)


## Construct and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
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
        
## Plot Unnormalized confusion matrix
class_names = np.array(['non-mutagen', 'mutagen'])
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

## Extracting dictionaries to make a full-record analysis
import re
import ast
indices = []
all_hyperparameters = []
scores = []
f = open('Grid_Search_Ouput.txt', 'r')
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

for i in range(len(all_hyperparameters)):
    all_hyperparameters[i]['index'] = indices[i]
    all_hyperparameters[i]['auc'] = scores[i]
    

## Writing the dictionaries into a csv file
import csv
with open('all_hyperparameters_2.csv', 'w', newline='') as csvfile:
    fieldnames = all_hyperparameters[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(all_hyperparameters)
    

#### Produce a Boxplot ####
import pandas as pd
data = pd.read_csv("all_hyperparameters_2.csv")