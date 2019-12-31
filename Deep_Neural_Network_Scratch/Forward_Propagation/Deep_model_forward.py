import numpy as np
from Deep_Neural_Network_Scratch.Test_File import *


def linear_forward(A, W, b):
	"""
	Implement the linear part of a layer's forward propagation.

	:param A:activations from previous layer (or input data)
	:param W:weights matrix
	:param b:bias vector
	:return:
	Z -- the input of the activation function, also called pre-activation parameter
	cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	Z = W.dot(A) + b
	assert (Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	:param A_prev: activations from previous layer
	:param W: weights matrix
	:param b: bias vector
	:param activation: the activation to be used in this layer: "sigmoid" or "relu"

	:return:
	A: the output of the activation function: post-activation layer
	cache: tuple containing "linear_cache" and "activation_cache",
	stored for computing the backward pass efficiently
	"""

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = 1 / (1 + np.exp(-Z)), Z

	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = np.maximum(0, Z), Z

	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache


def Deep_model_forward(X, parameters):
	"""
	Implement forward propagation for the
	[LINEAR -> ReLU]*(L-1)->LINEAR->SIGMOID
	:param X: data, numpy array of shape (input size, number of examples)
	:param parameters: output of initialize_parameters_deep()
	:return:
	AL: last post-activation value
	caches: list of caches containing:
	every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""

	caches = []
	A = X
	L = len(parameters) // 2  # number of layers in the neural network

	# Implement [LINEAR -> ReLU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
		caches.append(cache)

	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
	caches.append(cache)

	assert (AL.shape == (1, X.shape[1]))

	return AL, caches


if __name__ == '__main__':
	X, parameters = L_model_forward_test_case_2hidden()
	AL, caches = Deep_model_forward(X, parameters)
	print("AL = " + str(AL))
	print("Length of caches list = " + str(len(caches)))