# Keras Sequential Models on Chemical Property Predictions

We provide a quick-and-dirty neural network implementation to predict molecule's toxicity. This work follows the steps of a benchmark test that evaluates different machine learning models [(See 6th refrence in the project report)](https://github.com/kareemjeiroudi/molecules_and_ml/blob/main/doc/project_report.pdf). We take the appraoch of deep learning, optimize the model (hyperparameters set) using a grid search, and lastly make several statements about the best set of hyperparameters.

### Project Aim

A Sequential Neural Network operating on Morgan fingerprints of arbitrary bit size for chemical property prediction (e.g. mutgenity). Eventually, we compare our predictions against the officially classified structures by the Ames test [(See 7th reference in project report)](https://github.com/kareemjeiroudi/molecules_and_ml/blob/main/doc/project_report.pdf). We expect to see a match in the model's prediction and the Ames classification, when the model is best optimzied. 

### Quick Notes

This repository contains the data that's used for my semester project. Each python script in this repo accomplishes on step in the project workflow. Any insignificant data that's dropped here, will be moved to the directory draft. There is where I keep experimental files and all irrelevant code. That includes repetative scripts with slight modifications. Additionally, this repository is periodically getting cleaned up. I'm removing a lot of clutter every now and then.

Main, I use Jupyter Notebook to demonstrate my work from A-Z, but at some point in the project, I had to create independet python script that I could keep running on the cluster for longer time. Optimizing the hyperparameters takes obviously the longest, therefore do not attempt to run these scripts unless you running them on a powerful machine. I had to use a more powerful machine than my i5 laptop. In our case, the search took around **10 hours** when run serially on E5-**2640 v2 @ 2.00GHz CPU**. It also worth mentioning that for this purpose, one would desire to keep a log of all evaluated models in a separate file, which I did loosely. You can find that seperate file [here](https://github.com/kareemjeiroudi/molecules_and_ml/blob/main/Grid_Search_Ouput.txt).  _**Update (30. Sep. 2019):** I've written a new implementation that automatically takes care of that and writes the log to a file [`search_history.csv`](https://github.com/kareemjeiroudi/molecules_and_ml/blob/main/search_history.csv)._

### Environment Requirements

* Python 3, Numpy - best using [Anaconda](https://www.continuum.io/downloads) environments
* Either [Theano](http://deeplearning.net/software/theano/install.html) or [Tensorflow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) engine. In my case, I used Tensorflow.
* [RDkit](http://www.rdkit.org/docs/Install.html) - A handful of moldules for calculating or depicting chemical properties. Important for caculating the Morgan fingerprint (MF) as bit vectors and plotting molecules too.<br> The easiest way to [install RDkit is via using conda](https://www.rdkit.org/docs/Install.html) in the command line `conda install -c <https://conda.anaconda.org/rdkit> rdkit`, be aware of the dependecies of RDkit, plut the version you're installing, as there two versions: one for python 2; and the other for python 3. *The dependencies and other environment issues that you might encounter will be list here soon*.
  **Update (30. Sep. 2019):** I've had issues with newer versions of <font style="font-family: times; font-size: 14pt; font-">rdkit</font>, therefore, make sure you're running **`2018.09.3`** version.
* [scikit-learn](https://scikit-learn.org/stable/index.html) - A simple open source tool for data analysis and machine learning - comes in handy when working with molecular data. In our case, I used functions such as [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), to compute area under the receiver operating characteristic curve (**ROC AUC**) from prediction scores.

### Background Knowledge

#### Morgan Fingerprints (MF)

Is a handy way to emed molecular data for machine learning predictions. Results of MF is a bit-vector that has either 0 or 1 elements: 0 when a fingerprint (molecular structure of interest) is absent, and 1 when present. Since 2000 Morgan Fingerprints have become the standard of today's Molecule Isomorphism. Chemists use Morgan Fingerprints heavily in order to identify substructures in large molecules. These substructures, in turn, are of particular chemical features. In this project, we discuss molecular Mutagenicity, which is the type of molecules or sites than can induce mutating the DNA, and are, therefore, classified toxic (a.k.a Toxicophores). 

#### Mutagenicity

Mutagenicity in this very case, means the molecule's ability to induce DNA mutations. This is undesired property in drug design, when the drug targets humans DNA instead of bacterial DNA.

#### Keras Deep Networks

There have been attempts by several researchers to use Random Forest techniques as well as SVM in classifying this data of interest. Additionally, some literature compares these techinques and their classification accuracy to commercial machine learning solutions. The aim of this project is construct an Deep ANN to commit predictions on that data, and optimize the model using different strategies (e.g. Grid Search or Bayesian Optimization). Machine learning algorithms exclusively derive their knowledge from the training data. The power of binary classifiers is desired for this classification problem, but is higly sophisticated at the same time. It can only be triggered by building and evaluating different models. In that regard, we dicuss the process of Hyperparamter Tuning in more details in a sepearate section.

-------

### Getting Started

To follow along the whole procedure, please take a look on the [project's report](https://github.com/kareemjeiroudi/molecules_and_ml/blob/main/doc/project_report.pdf). Comments and descriptions of methods are provided within the report: [Keras Sequential Models on Chemical Property Predictions](report/Keras_Sequential_Models_on_Chemical_Property_Predictions.pdf)



## Update 05. April

- [x] ~~Soon I'll be making a well-documented Jupyter Notebook from A to Z to summarize all of the work and get rid of all these separate scripts. Just bare with me for now, if you're trying to access these project files. I know the repository is cluttered up at the moment~~ - done!
- [ ] Re-optimze the model using Bayesian inference this time. This has two main adventages over the previous grid search:  a) it's much more time-saving; b) it returns more potetionally interesting hyperparameter settings that worth comparing. At the time, the first report was made, analysis of the best-scoring model were rather uninformative. In order to avoid this error, and get more promising statistical analysis, I've set out to optimize the model's using Bayesian Inference this time. 