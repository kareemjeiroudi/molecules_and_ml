#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:30:44 2019

@author: kareem
@Date: 09.03.2019
@Title: Last script in my project to load trained model and evaluate it against
    a test set.
"""
### Loading Model weights and architecture ####
import keras
from keras.models import model_from_json
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
## load json and create model
jf = open('analysis_best_model/best_model.json', 'r')
best_model = model_from_json(jf.read())
jf.close()
## load weights into new model
best_model.load_weights("analysis_best_model/best_model.h5")
## Compile the model after loading
best_model.compile(optimizer=SGD(lr=0.1), loss='binary_crossentropy', metrics=[binary_accuracy])



#### Read Test data ####
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
fingerprints = []
y_true = []
info={}
with open('data/test.csv', 'r') as f:
    reader = csv.reader(f)
    i=0
    for row in reader:
        if i==0:
            i+=1
            continue
        ## read molecule from SMILES
        m = Chem.MolFromSmiles(row[1])
        ## Calculate fingerprint accordingly
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048, bitInfo=info)
        ## Append fp into fingerprints
        fingerprints.append(fp)
        y_true.append(int(row[2]))



#### Standard Scaling ####
from sklearn.preprocessing import StandardScaler
import numpy as np
#Scale fingerprints to unit variance and zero mean
scaler = StandardScaler()
X_test = scaler.fit_transform(np.array(fingerprints))
y_true = np.array(y_true)



#### Predicting on the Test Set ####
y_prob = best_model.predict(X_test)
y_pred = best_model.predict_classes(X_test)
## if you didn't load the model from the harddisk, use this
y_prob = classifier.predict(X_test)
y_pred = classifier.predict_classes(X_test)

### Evaluating Model's predictions ####
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
auc = roc_auc_score(y_true, y_prob)
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
plot_roc_curve(fpr, tpr, auc) # load this function!
## Construct and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
class_names = np.array(['non-mutagen', 'mutagen'])
plot_confusion_matrix(cm, classes=class_names, # load this function!
                      title='Confusion matrix, without normalization')