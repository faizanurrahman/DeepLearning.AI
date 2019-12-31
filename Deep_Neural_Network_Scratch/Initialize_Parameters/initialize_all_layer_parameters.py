import numpy as np
from Deep_Neural_Network_Scratch.Test_File import *


def initialize_parameters_deep(layer_dims, initialization='random'):
	"""
	Arguments:
	layer_dims -- list containing the dimensions of each layer in our network

	Returns:
	parameters -- dictionary containing parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""

	np.random.seed(3)  # for testing purpus
	parameters = {}
	L = len(layer_dims)  # number of layers in the network

	for l in range(1, L):
		# initialize parameters for each layer
		if initialization == 'random':
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		elif initialization == 'he':
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * (np.sqrt(2/layer_dims[l-1]))
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		elif initialization == 'xavier':
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * (np.sqrt(1/layer_dims[l - 1]))
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
		else:
			print("Please Check the parameter initialization type: ")
		# check shape of parameters
		assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
		assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

	return parameters


if __name__ == '__main__':
	parameters = initialize_parameters_deep([5, 4, 3], initializaiton = 'xavier')
	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
