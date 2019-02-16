%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from keras import backend as K
from keras.models import Sequential
from keras.layers import InputLayer, Input, Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import load_model

## pip install h5py scikit-optimize
## once you have that installed, you need to install a bunch of thing sfrom scikit optimize.
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

## Before starting the session
print(tf.__version__)
print(tf.keras.__version__)
print(skopt.__version__)
## to set up this search space, I first need to define the search space dimension, what parameters are we gonna explore.
## for each of the parameters, we define a dimension explicitly

## the learning rate is any real number between 0.000001 and 0.1. But the seraching is done not in bounds.
## 'log-uniform' specifies how the trasformation(updates) of these values is 
learning_rate_dim = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
## The number of alyers on the other hand is explored in bounds, increments are done using integers
dense_layers_dim = Integer(low=1, high=5, name='dense_layers')
## We'll also different number of nodes in a layer
nodes_dim = Integer(low=5, high=512, name='nodes')
## finally we have a Categorical dimension, this needs to be specified explicitly, because scikit-learn
## isn't gonna generate some randomly for you
activation_dim = Categorical(categories=['relu', 'sigmoid'], name='activation')
## combine all the parameters into a list, so that we can pass it to a function
dimensions = [learning_rate_dim,
			dense_layers_dim,
			nodes_dim,
			activation_dim]


## to kick off, it's helpful to start the serach using a set of hyperparameters that we
## intuitively know performes well
## these default parameters aren't horrible, but they don't perform great either
default_parameters = [1e-5, 1, 16, 'relu']



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
def create_model(learning_rate, dense_layers, nodes, activation):
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
	## Note that the input-shape must be a tupe containing the data-size.
	model.add(InputLayer(input_shape=(here,)))
	## Needful only in case of convolutional layers.
	# model.add(Reshape(img_shape_full))
	for i in range(dense_layers):
		## Name each layer, because Keras should give them unique names.
		name = 'layer_dense_{0}.format(i+1)'
		## Add these fully-connected layers to the model.
		model.add(Dense(nodes, activation=activation, name=name))

	## Last output layer with softmax-activation.
	## Used heavily for classification.
	model.add(Dense(num_classes, activation='softmax'))

	optimizer = Adam(lr=learning_rate)
	## Compile the model
	model.Compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

	return model

 
## Before we start training any model, let's first save the path where we'll store the best-performing model.
best_model_path = '19_best_model.keras'
## A global variable to keep track of the best obtained accuracy.
best_accuracy = 0.0

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, dense_layers, nodes, activation):
	"""
	"""
	# Print the selected hyperparameters.
	print('learning rate: {0:.1e}'.format(learning_rate))
	print('num_dense_layers:', dense_layers)
	print('num_nodes:', nodes)
	print('activation:', activation)
	print()
	## Create the neural network with these hyperparameters.
	model = create_model(learning_rate, dense_layers, nodes, activation)
	## Create log files for the model.
	callback_log = TensorBoard(
		log_dir=log_dir,
		histogram_freq=0,
		batch_size=32,
		write_graph=True,
		write_grads=False,
		write_images=False)
	## Use Keras to train the model.
	history = model.fit(x=train_data,
		y=train_labels,
		epochs=3,
		batch_size=128,
		validation_data=validation_data,
		callbacks=[callback_log])
	## Get the classification accuracy on the validation set after the last training epoch.
	accuracy = history.history['val_acc'][-1]

	## Print the calssification accuracy.
	print()
	print("Accuracy: {0:.2%}".format(accuracy))
	print()

	## Save the model if it improves on the best-found performance.
	## We use the global keyword so we update the variable outside of this function.
	global best_accuracy
	if accuracy > best_accuracy:
		## Save the new model to harddisk.
		model.save(path_best_model)
		## Update the classification accuracy.
		best_accuracy = accuracy

	## Delete the Keras model with these heyper parameters from memory.
	## Also clear the session.
	del model
	K.clear_session()

	return -accuracy

## Now we run our fitness function with the default hyperparameters that we set earlier.
## That's the reason for the @ annotation 
fitness(x=default_parameters)

serach_result = gp_minimize(func=fitness,
	dimensions=dimensions,
	acq_func='EI', # Expected Improvement.
	n_calls=40,
	x0=default_parameters)

## Plot the progress of the optimizer.
plot_convergence(serach_result)
print("Best serach results:")
print(serach_result.x)
space = serach_result.space
print(space.point_to_dict(serach_result.x))
print("Lowest fitness value:")
print(search_result.fun)
print(sorted(zip(search_result.func_vals, search_result.x_iters)))

