# Keras Sequential Models on Chemical Property Predictions

### Project Aim

A Sequential Neural Network operating on Morgan fingerprints of arbitrary bit size for chemical property prediction (e.g. mutgenity).

### Quick Note

This repository contains the data that's used for my bachelor project. Any insignificant data that's dropped here, will be moved to the directory draft. There is where I keep experimental files and all irrelevant code.
Usually submitibble files are stored as Jupyter Notebook format. With that being said any python files `.py` you find here, don't serve the project directly.

###Requirements:

* Python 3, Numpy - best using [Anaconda](https://www.continuum.io/downloads) environments

* Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) engine. In my case, I used Tensorflow.

* [RDkit](http://www.rdkit.org/docs/Install.html) - A handful of moldules for calculating or depicting chemical properties.<br>The easiest way to [install RDkit is via using conda](https://www.rdkit.org/docs/Install.html) in the command line `conda install -c <https://conda.anaconda.org/rdkit> rdkit`, be aware of the dependecies of RDkit, plut the version you're installing, as there two versions: one for python 2; and the other for python 3. *The dependencies and other environment issues that you might encounter will be list here soon*.
* [scikit-learn](https://scikit-learn.org/stable/index.html) - A simple open source tool for data analysis - comes in handy when working with molecular data. In our case, I used the function [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), for compute area under the receiver operating characteristic curve (**ROC AUC**) from prediction scores.



-------

To follow along the whole procedure, please take a look on the project's report. Comments and descriptions of methods are provided within the report: [Keras Sequential Models on Chemical Property Predictions](report/Keras_Sequential_Models_on_Chemical_Property_Predictions.pdf)