import numpy as np


def initialize_velocity(parameters):
	"""
	Initializes the velocity as a python dictionary with:
			- keys: "dW1", "db1", ..., "dWL", "dbL"
			- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

	:param parameters: python dictionary containing your parameters.
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	:return: v: python dictionary containing the current velocity.
					v['dW' + str(l)] = velocity of dWl
					v['db' + str(l)] = velocity of dbl
	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	v = {}

	# Initialize velocity
	for l in range(L):
		v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape))
		v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape))

	return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	"""
	Update parameters using Momentum

	:param parameters: python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	:param grads: python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	:param v: python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	:param beta: the momentum hyperparameter, scalar
	:param learning_rate: the learning rate, scalar
	:return: parameters: python dictionary containing your updated parameters
	:return: v: python dictionary containing your updated velocities
	"""

	L = len(parameters) // 2  # number of layers in the neural networks

	# Momentum update for each parameter
	for l in range(L):
		# compute velocities
		v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
		v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
		# update parameters
		parameters["W" + str(l + 1)] -= learning_rate * v["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] -= learning_rate * v['db' + str(l + 1)]

	return parameters, v

