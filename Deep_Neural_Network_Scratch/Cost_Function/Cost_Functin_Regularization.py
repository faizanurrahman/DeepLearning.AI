import numpy as np

from Deep_Neural_Network_Scratch.Cost_Function.Cost_Fuction import compute_cost


def compute_cost_with_regularization(AL, Y, lambd):
	"""
	Implement the cost function with L2 regularization. See formula (2) above.

	Arguments:
	AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	parameters -- python dictionary containing parameters of the model

	Returns:
	cost - value of the regularized loss function (formula (2))
	"""
	m = Y.shape[1]

	cost = (-1 / m) * np.sum(Y.dot(np.log(AL.T)) + (1 - Y).dot(np.log(1 - AL.T)))
	cost = np.squeeze(cost)

	assert (cost.shape == ())

	cross_entropy_cost = cost  # This gives you the cross-entropy part of the cost

	L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

	cost = cross_entropy_cost + L2_regularization_cost

	return cost
