# Keras Sequential Models on Chemical Property Predictions

### Project Aim

A Sequential Neural Network operating on Morgan fingerprints of arbitrary bit size for chemical property prediction (e.g. mutgenity).

### Quick Note

This repository contains the data that's used for my semester project. Each python script in this repo accomplishes on step in the project workflow. Any insignificant data that's dropped here, will be moved to the directory draft. There is where I keep experimental files and all irrelevant code. That includes repetative scripts with slight modifications.
Initially, I used Jupyter Notebook to indicate submittable file, but later I had to create independet python script that I could keep running on the cluster for longer time. Training the network took obviously the longest, therefore I had to use a more powerful machine than my i5 laptop. With that being said any python files `.py` you find here constitue to a smaller part of this project.

### Requirements:

* Python 3, Numpy - best using [Anaconda](https://www.continuum.io/downloads) environments

* Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) engine. In my case, I used Tensorflow.

* [RDkit](http://www.rdkit.org/docs/Install.html) - A handful of moldules for calculating or depicting chemical properties.<br>The easiest way to [install RDkit is via using conda](https://www.rdkit.org/docs/Install.html) in the command line `conda install -c <https://conda.anaconda.org/rdkit> rdkit`, be aware of the dependecies of RDkit, plut the version you're installing, as there two versions: one for python 2; and the other for python 3. *The dependencies and other environment issues that you might encounter will be list here soon*.
* [scikit-learn](https://scikit-learn.org/stable/index.html) - A simple open source tool for data analysis - comes in handy when working with molecular data. In our case, I used the function [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), to compute area under the receiver operating characteristic curve (**ROC AUC**) from prediction scores.

### Introduction

Since 2000 Morgan Fingerpritns have become the standard of today's Molecule Isomorphism. Chemists use Morgan Fingerprints heavily in order to identify substructures in large molecules. These substructures, in turn, are of particular chemical features. In this project, we discuss molecular Mutagenicity, which is the type of molecules or sites than can induce mutating the DNA, and are, therefore, classified toxic. 

#### Mutagenicity 

#### Calculating the Fingerprints

### Methods

#### Keras Deep Networks

There have been attempts by several researchers to use Random Forest techniques as well as SVM in classifying this data of interest. Additionally, some literature compares these techinques and their classification accuracy to commercial machine learning solutions. The aim of this project is construct an Deep ANN to commit predictions on that data, and optimize the model using different strategies (e.g. Grid Search and Bayesian Optimization). Machine learning algorithms exclusively derive their knowledge from the training data. The power of binary classifiers is desired for this classification problem, but is higly sophisticated at the same time. It can only be triggered by building and evaluating different models. In that regard, we dicuss the process of Hyperparamter Tuning in more details in a sepearate section.

-------

To follow along the whole procedure, please take a look on the project's report. Comments and descriptions of methods are provided within the report: [Keras Sequential Models on Chemical Property Predictions](report/Keras_Sequential_Models_on_Chemical_Property_Predictions.pdf)
