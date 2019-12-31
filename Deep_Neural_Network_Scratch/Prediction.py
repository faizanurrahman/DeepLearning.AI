import numpy as np

from Deep_Neural_Network_Scratch.Forward_Propagation.Deep_model_forward import Deep_model_forward


def predict(X, y, parameters):
	"""
	This function is used to predict the results of a  n-layer neural network.

	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model

	Returns:
	p -- predictions for the given dataset X
	"""

	m = X.shape[1]
	p = np.zeros((1, m), dtype=np.int)

	# Forward propagation
	a3, caches = Deep_model_forward(X, parameters)

	# convert probas to 0/1 predictions
	for i in range(0, a3.shape[1]):
		if a3[0, i] > 0.5:
			p[0, i] = 1
		else:
			p[0, i] = 0

	# print results

	# print ("predictions: " + str(p[0,:]))
	# print ("true labels: " + str(y[0,:]))
	print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

	return p
