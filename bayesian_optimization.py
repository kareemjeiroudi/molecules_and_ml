# %matplotlib inline
import tensorflow as tf
#import tensorflow.keras as K
import numpy as np
import math
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt

# from keras import backend as K
from keras.models import Sequential, load_model
# from K.models import Sequential, load_model
from keras.layers import InputLayer, Input, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.backend import clear_session
## pip install h5py scikit-optimize
## once you have that installed, you can run the following code.
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
matplotlib.use('GTKAgg')
from skopt.plots import plot_convergence
matplotlib.use('GTKAgg')
from skopt.plots import plot_objective, plot_evaluations
matplotlib.use('GTKAgg')
import csv
from timeit import default_timer as timer

#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from sklearn.metrics import roc_auc_score ## Computer Area Under the Curve
from datetime import datetime ## time the Optimization time

## Load Datset
train_samples = np.loadtxt("data/train_samples.txt", delimiter=' ', comments='# ', encoding=None)
train_labels = np.loadtxt("data/train_labels.txt", delimiter=' ', comments='# ', encoding=None)
valid_samples = np.loadtxt("data/valid_samples.txt", delimiter=' ', comments='# ', encoding=None)
valid_labels = np.loadtxt("data/valid_labels.txt", delimiter=' ', comments='# ', encoding=None)

## To set up this search space, I first need to define the search space dimension, what parameters are we gonna explore.
## for each of the parameters, we define a dimension explicitly
##
## The learning rate is any real number between 0.000001 and 0.1. But the seraching is done not in bounds.
## 'log-uniform' specifies how the trasformation(updates) of these values is 
learning_rate_dim = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
## The number of alyers on the other hand is explored in bounds, increments are done using integers
dense_layers_dim = Integer(low=1, high=5, name='dense_layers')
## We'll also different number of nodes in a layer
nodes_dim = Integer(low=5, high=512, name='nodes')
## Finally we have a Categorical dimension, this needs to be specified explicitly, because scikit-learn
## isn't gonna generate some randomly for you
activation_dim = Categorical(categories=['relu', 'sigmoid'], name='activation')
## Combine all the parameters into a list, so that we can pass it to a function
dimensions = [learning_rate_dim,
			dense_layers_dim,
			nodes_dim,
			activation_dim]


## To kick off, it's helpful to start the serach using a set of hyperparameters that we
## intuitively know performes well
## These default parameters aren't horrible, but they don't perform great either
default_parameters = [1e-5, 1, 16, 'relu']


## To log the performance of the model
def log_dir_name(learning_rate, dense_layers, nodes, activation):
	"""
	Creates a directory named after the set of hyperparameters that was recently selected. A helper function
	to log the results of training every constructed model.
	"""	
	# the dir-name for the TensorBoard log-dir
	s = "./2_logs/lr_{0:.0e}_layers{1}_nodes{2}_{3}/"
	log_dir = s.format(learning_rate, dense_layers, nodes, activation)

	return log_dir


## This funcion is copied from my previous solution on Grid SearchCV
def create_model(learning_rate, dense_layers, nodes, activation, dropout_rate=0.1):
	"""
	A helper function for the classifier to help construct a model after each run.

	learing_rate:	Learning-rate for the optimizer.
	dense_layer: 	Number of dense layers for the sequentail model
	nodes:			Number of nodes in each inner dense layer.
	activation:		Activation function for all layers.
	Additionally, we can improve on this function by adding a separate activation for
	the output layer.
	"""
	model = Sequential()
	global train_samples
	## Input-shape must be a tuple without the batch size.
	input_shape = (1,) + train_samples.shape
	model.add(InputLayer(input_shape=(len(train_samples[0]),)))
	## Needful only in case of convolutional layers.
	# model.add(Reshape(img_shape_full))
	for i in range(dense_layers):
		## Name each layer, because Keras should give them unique names.
		name = 'layer_dense_{0}'.format(i+1)
		## Add these fully-connected layers to the model.
		model.add(Dense(nodes, activation=activation, name=name))
		model.add(Dropout(dropout_rate))

	## Last output layer with softmax-activation.
	## Used heavily for classification.
	model.add(Dense(1, activation='sigmoid'))

	optimizer = Adam(lr=learning_rate)
	## Compile the model
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

	return model

 
## Before we start training any model, let's first save the path where we'll store the best-performing model.
best_model_path = '19_best_model.keras'
## A global variable to keep track of the best obtained accuracy.
best_auc = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dense_layers, nodes, activation):
	"""
	"""
	# Print the selected hyperparameters.
	print('learning rate: {0:.1f}'.format(learning_rate))
	print('num_dense_layers:', dense_layers)
	print('num_nodes:', nodes)
	print('activation:', activation)
	print("")
	## Create the neural network with these hyperparameters.
	model = create_model(learning_rate, dense_layers, nodes, activation)
	## Create log files for the model.
	## Not important for now!
	# callback_log = TensorBoard(
	# 	log_dir=log_dir,
	# 	histogram_freq=0,
	# 	batch_size=32,
	# 	write_graph=True,
	# 	write_grads=False,
	# 	write_images=False)
	## Use Keras to train the model.
	history = model.fit(x=train_samples,
		y=train_labels,
		epochs=10,
		batch_size=int(4010/4))
		#callbacks=[callback_log])
	## Get the classification accuracy on the validation set after the last training epoch.
	# accuracy = history.history['val_acc'][-1]
	predictions = model.predict(valid_samples)
	auc = roc_auc_score(valid_labels, predictions)
	## Print the calssification accuracy.
	print('')
	print("AUC = : {0:.2%}".format(auc))
	print('')

	## Save the model if it improves on the best-found performance.
	## We use the global keyword so we update the variable outside of this function.
	global best_auc
	if auc > best_auc:
		## Save the new model to harddisk.
		model.save(best_model_path)
		## Update the classification accuracy.
		best_auc = auc

	## Delete the Keras model with these heyper parameters from memory.

	## Also clear the session.
	del model
#    tf.keras.clear_session()
	clear_session()

	return -auc

## Now we run our fitness function with the default hyperparameters that we set earlier.
## That's the reason for the @ annotation 
fitness(x=default_parameters)

search_result = gp_minimize(func=fitness,
	dimensions=dimensions,
	acq_func='EI', # Expected Improvement.
	n_calls=40,
	x0=default_parameters)

## Report Result of the optimizer.
print("Best serach results:")
print(search_result.x)
print(search_result.space)
print("Lowest fitness value:")
print(search_result.fun)
zipped = sorted(zip(search_result.func_vals, search_result.x_iters))
print(zipped)

## Write sorted results to csv file for exporting
of = open('output_bayesian_optimization.csv', 'w')
header="Fit Value; Learning Rate; Dense Layers; Num. Neurons; Activation\n"
of.write(header)
for i in zipped:
    row = "{0}; {1}; {2}; {3}; {4};\n".format(i[0], i[1][0], i[1][1], i[1][2], i[1][3])
    of.write(row)
of.close()

## Plot results of optimizer
dim_names = ['learning_rate', 'dense_layers', 'nodes', 'activation']
plot_objective(search_result, dimensions=dim_names)
plot_evaluations(search_result)



