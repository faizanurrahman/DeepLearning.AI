import numpy as np
import matplotlib.pyplot as plt
from Deep_Neural_Network_Scratch.Backward_Propagation.Deep_model_backward import Deep_model_backward
from Deep_Neural_Network_Scratch.Cost_Function.Cost_Fuction import compute_cost
from Deep_Neural_Network_Scratch.Forward_Propagation.Deep_model_forward import Deep_model_forward
from Deep_Neural_Network_Scratch.Initialize_Parameters.initialize_all_layer_parameters import initialize_parameters_deep
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_adam import initialize_adam, \
	update_parameters_with_adam
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_gd import update_parameters_with_gd
from Deep_Neural_Network_Scratch.Optimisation.update_parameters_with_momentum import initialize_velocity, \
	update_parameters_with_momentum


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
	"""
	Creates a list of random minibatches from (X, Y)

	:param X: input data, of shape (input size, number of examples)
	:param Y: true "label" vector
	:param mini_batch_size: size of the mini-batches, integer
	:param seed:
	:return: mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
	"""

	np.random.seed(seed)  # To make your "random" minibatches the same as ours
	m = X.shape[1]  # number of training examples
	mini_batches = []

	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1, m))

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(
		m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size:mini_batch_size * (k + 1)]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size:mini_batch_size * (k + 1)]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, :-(m % mini_batch_size)]
		mini_batch_Y = shuffled_Y[:, :-(m % mini_batch_size)]

		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches


def model_mini_batches(X, Y, layers_dims, optimizer, initialization='random', learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
	"""
	N-layer neural network model which can be run in different optimizer modes.
	:param: initialization: weight parameter initialization
	:param: X: input data, of shape (2, number of examples)
	:param: Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	:param: layers_dims: python list, containing the size of each layer
	:param: learning_rate: the learning rate, scalar.
	:param: mini_batch_size: the size of a mini batch
	:param: beta: Momentum hyperparameter
	:param: beta1: Exponential decay hyperparameter for the past gradients estimates
	:param: beta2: Exponential decay hyperparameter for the past squared gradients estimates
	:param: epsilon: hyperparameter preventing division by zero in Adam updates
	:param: num_epochs: number of epochs
	:param: print_cost: True to print the cost every 1000 epochs
	:return: parameters -- python dictionary containing your updated parameters
	"""

	L = len(layers_dims)  # number of layers in the neural networks
	costs = []  # to keep track of the cost
	t = 0  # initializing the counter required for Adam update
	seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
	m = X.shape[1]  # number of training examples

	# Initialize parameters dictionary.
	parameters = initialize_parameters_deep(layers_dims, initialization=initialization)

	# Initialize the optimizer
	if optimizer == "gd":
		pass  # no initialization required for gradient descent
	elif optimizer == "momentum":
		v = initialize_velocity(parameters)
	elif optimizer == "adam":
		v, s = initialize_adam(parameters)

	# Optimization loop
	for i in range(num_epochs):

		# Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
		seed = seed + 1
		minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		cost_total = 0

		for minibatch in minibatches:

			# Select a minibatch
			(minibatch_X, minibatch_Y) = minibatch

			# Forward propagation
			a3, caches = Deep_model_forward(minibatch_X, parameters)

			# Compute cost and add to the cost total
			cost_total += compute_cost(a3, minibatch_Y)

			# Backward propagation
			grads = Deep_model_backward(minibatch_X, minibatch_Y, caches)

			# Update parameters
			if optimizer == "gd":
				parameters = update_parameters_with_gd(parameters, grads, learning_rate)
			elif optimizer == "momentum":
				parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer == "adam":
				t = t + 1  # Adam counter
				parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

		cost_avg = cost_total / m

		# Print the cost every 1000 epoch
		if print_cost and i % 1000 == 0:
			print("Cost after epoch %i: %f" % (i, cost_avg))
		if print_cost and i % 100 == 0:
			costs.append(cost_avg)

	# plot the cost
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('epochs (per 100)')
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()

	return parameters
