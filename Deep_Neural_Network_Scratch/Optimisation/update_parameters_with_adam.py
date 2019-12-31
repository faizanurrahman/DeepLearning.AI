import numpy as np


def initialize_adam(parameters):
	"""
	Initializes v and s as two python dictionaries with:
				- keys: "dW1", "db1", ..., "dWL", "dbL"
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.





	:param parameters: python dictionary containing your parameters.
					parameters["W" + str(l)] = Wl
					parameters["b" + str(l)] = bl
	:return: v: python dictionary that will contain the exponentially weighted average of the gradient.
					v["dW" + str(l)] = ...
					v["db" + str(l)] = ...
	:return: s: python dictionary that will contain the exponentially weighted average of the squared gradient.
					s["dW" + str(l)] = ...
					s["db" + str(l)] = ...
	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	v = {}
	s = {}

	# Initialize v, s. Input: "parameters". Outputs: "v, s".
	for l in range(L):
		v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
		v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))
		s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape))
		s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape))

	return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
	"""
	Update parameters using Adam

	:param parameters: python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	:param grads: python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	:param v: Adam variable, moving average of the first gradient, python dictionary
	:param s: Adam variable, moving average of the squared gradient, python dictionary
	:param t: Moving average variable amount of consider previous data, used for bias correction
	:param learning_rate: the learning rate, scalar.
	:param beta1: Exponential decay hyperparameter for the first moment estimates
	:param beta2: Exponential decay hyperparameter for the second moment estimates
	:param epsilon: hyperparameter preventing division by zero in Adam updates
	:return: parameters: python dictionary containing your updated parameters
	:return: v: Adam variable, moving average of the first gradient, python dictionary
	:return s: Adam variable, moving average of the squared gradient, python dictionary
	"""

	L = len(parameters) // 2  # number of layers in the neural networks
	v_corrected = {}  # Initializing first moment estimate, python dictionary
	s_corrected = {}  # Initializing second moment estimate, python dictionary

	# Perform Adam update on all parameters
	for l in range(L):
		# Moving average of the gradients.
		v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
		v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

		# Compute bias-corrected first moment estimate.
		v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
		v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

		# Moving average of the squared gradients.
		s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)], 2)
		s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)], 2)

		# Compute bias-corrected second raw moment estimate.
		s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
		s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

		# Update parameters.
		parameters["W" + str(l + 1)] -= learning_rate * (
				v_corrected["dW" + str(l + 1)] / (np.power(s_corrected["dW" + str(l + 1)] + epsilon, 0.5)))
		parameters["b" + str(l + 1)] -= learning_rate * (
				v_corrected["db" + str(l + 1)] / (np.power(s_corrected["db" + str(l + 1)] + epsilon, 0.5)))

	return parameters, v, s
