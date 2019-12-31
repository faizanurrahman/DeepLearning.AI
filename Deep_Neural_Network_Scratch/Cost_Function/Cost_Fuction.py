import numpy as np
from Deep_Neural_Network_Scratch.Test_File import *


def compute_cost(AL, Y):
	"""
	Implement the cross-entropy cost function.

	:param AL: probability vector corresponding to your label predictions, shape (1, number of examples)
	:param Y: true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
	:return: cost -- cross-entropy cost

	"""

	m = Y.shape[1]

	cost = (-1 / m) * np.sum(Y.dot(np.log(AL.T)) + (1 - Y).dot(np.log(1 - AL.T)))
	cost = np.squeeze(cost)

	assert (cost.shape == ())

	return cost


if __name__=='__main__':
	Y, AL = compute_cost_test_case()

	print("cost = " + str(compute_cost(AL, Y)))