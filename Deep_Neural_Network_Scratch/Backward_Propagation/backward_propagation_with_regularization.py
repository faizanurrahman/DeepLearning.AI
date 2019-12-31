import numpy as np

#from Deep_Neural_Network_Scratch.Backward_Propagation.Deep_model_backward import linear_activation_backward


def linear_backward(dZ, cache, lambd):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	:param dZ: Gradient of the cost with respect to the linear output (of current layer l)
	:param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
	:return:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambd / m) * W
	db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T, dZ)

	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev, dW, db


def linear_activation_backward(dA, cache, lambd, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer
	:param dA:post-activation gradient for current layer l
	:param cache:tuple of values (linear_cache, activation_cache)
	:param activation:the activation to be used in this layer, stored as a text string: "sigmoid" or "relu".

	:return:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""

	linear_cache, activation_cache = cache

	if activation == "relu":

		Z = activation_cache
		dZ = np.array(dA, copy=True)
		dZ[Z <= 0] = 0

		assert (dZ.shape == Z.shape)

		# dZ = relu_backward(dA, activation_cache)

		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

	elif activation == "sigmoid":

		s = 1 / (1 + np.exp(-activation_cache))
		dZ = dA * s * (1 - s)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

	return dA_prev, dW, db


def backward_propagation_with_regularization(X, Y, caches, lambd):
	"""
		Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

		:param AL: probability vector, output of the forward propagation (Deep_model_forward())
		:param Y: true "label" vector
		:param caches: list of caches containing:
					every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
					the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
		:param lambd: regularization parameter
		:return: A dictionary with the gradients
					grads["dA" + str(l)] = ...
					grads["dW" + str(l)] = ...
					grads["db" + str(l)] = ...
		"""
	AL = X
	grads = {}
	L = len(caches)  # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

	# Initializing the back propagation
	# gradient of last output layer with respect to final loss
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	# Lth layer (SIGMOID -> LINEAR) grads
	current_cache = caches[L - 1]
	grads["dA" + str(L - 1)], \
	grads["dW" + str(L)], \
	grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, lambd, activation='sigmoid')

	# Loop from l=L-2 to l=0, for other layers grads
	for l in reversed(range(L - 1)):
		# lth layer: (ReLU -> LINEAR) gradients.
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, lambd, activation='relu')
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads
