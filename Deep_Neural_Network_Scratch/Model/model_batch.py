import matplotlib.pyplot as plt
from Deep_Neural_Network_Scratch.Initialize_Parameters.initialize_all_layer_parameters import *
from Deep_Neural_Network_Scratch.Forward_Propagation.Deep_model_forward import *
from Deep_Neural_Network_Scratch.Backward_Propagation.Deep_model_backward import *
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_adam import *
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_gd import *
from Deep_Neural_Network_Scratch.Cost_Function.Cost_Fuction import compute_cost
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_momentum import *


def model(X, Y, layer_dims, optimizer, learning_rate=0.01, num_iterations=15000, beta=0.9, beta1=0.9, beta2=0.999, epsilon = 1e-8,  print_cost=True, initialization="random"):
	"""
	Implements a layers_dims-layer neural network: (L-1)*(LINEAR->ReLU->)->LINEAR->SIGMOID.

	:param epsilon:
	:param beta2:
	:param beta1:
	:param beta:
	:param X: input data, of shape (number of feature, number of examples)
	:param Y: true "label" vector
	:param optimizer: type of optimization algorithm
	:param layer_dims: layer dimensions including input feature
	:param learning_rate: learning rate for gradient descent
	:param num_iterations: number of iterations to run gradient descent
	:param print_cost: if True, print the cost every 1000 iterations
	:param initialization: flag to choose which initialization to use ("random" or "he" or "xavier)
	:return: parameters: parameters learnt by the model
	"""

	grads = {}
	costs = []  # to keep track of the loss
	m = X.shape[1]  # number of examples
	layers_dims = layer_dims

	# Initialize parameters dictionary.
	parameters = initialize_parameters_deep(layers_dims, initialization=initialization)

	# Initialize the optimizer
	if optimizer == "gd":
		pass  # no initialization required for gradient descent
	elif optimizer == "momentum":
		v = initialize_velocity(parameters)
	elif optimizer == "adam":
		v, s = initialize_adam(parameters)

	# Loop (gradient descent)
	for i in range(0, num_iterations):

		# Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
		a3, cache = Deep_model_forward(X, parameters)

		# Loss
		cost = compute_cost(a3, Y)

		# Backward propagation.
		grads = Deep_model_backward(X, Y, cache)

		# Update parameters
		if optimizer == "gd":
			parameters = update_parameters_with_gd(parameters, grads, learning_rate)
		elif optimizer == "momentum":
			parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
		elif optimizer == "adam":
			t = t + 1  # Adam counter
			parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

		# Print the loss every 1000 iterations
		if print_cost and i % 1000 == 0:
			print("Cost after iteration {}: {}".format(i, cost))
			costs.append(cost)

	# plot the loss
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters
