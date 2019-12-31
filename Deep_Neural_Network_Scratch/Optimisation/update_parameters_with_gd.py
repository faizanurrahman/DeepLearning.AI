import numpy as np
from Deep_Neural_Network_Scratch.Test_File import *


def update_parameters_with_gd(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent

	:param parameters: python dictionary containing your parameters
	:param grads: python dictionary containing your gradients, output of Deep_model_backward
	:param learning_rate: how much to update the parameters
	:return:
	python dictionary containing your updated parameters
				parameters["W" + str(l)] = ...
				parameters["b" + str(l)] = ...
	"""

	L = len(parameters) // 2  # number of layers in the neural network

	# Update rule for each parameter.

	for l in range(L):
		parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]

	return parameters


if __name__ == '__main__':
	parameters, grads = update_parameters_test_case()
	parameters = update_parameters_with_gd(parameters, grads, 0.1)

	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))