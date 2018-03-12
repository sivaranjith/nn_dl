"""
This is file contains network class which repersents the neuralnetwork and does actions like feedforward and gradient update
"""

import numpy as np
import random

class Network(object):

	def __init__(self, neurons_per_layer):
		self.no_of_layers = len(neurons_per_layer)
		self.biases = [np.random.randn(col, 1) for col in neurons_per_layer[1:]]
		self.weights = [np.random.randn(col, row) for row, col in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]

	def feedforward(self, mini_batch, z_vals = None, a_vals = None):
		for b, w in zip(self.biases, self.weights):
			z = np.matmul(w, mini_batch) + b
			mini_batch = sigmoid(z)
			if z_vals != None:
				z_vals.append(z)
				a_vals.append(mini_batch)
		return mini_batch

	def train(self, training_data, no_of_epochs, mini_batch_size, learning_rate, validation_data = None):
		training_data_count = len(training_data)
		for curr_epoch_num in range(no_of_epochs):
			random.shuffle(training_data)
			mini_batch_pairs = [training_data[start:start + mini_batch_size] for start in range(0, training_data_count, mini_batch_size)]

			for mini_batch_pair in mini_batch_pairs:
				self.update_mini_batch(mini_batch_pair, learning_rate)

			if validation_data != None:
				print(curr_epoch_num, self.evaluate(validation_data), len(validation_data))
			else:
				print("no test data available")

	def update_mini_batch(self, mini_batch_pair, learning_rate):
		mini_batch = []
		labels = []

		for data, label in mini_batch_pair:
			mini_batch.append(data)
			labels.append(label)
		mini_batch = np.array(mini_batch)
		labels = np.array(labels)

		z_vals = []
		a_vals = [mini_batch]

		self.feedforward(mini_batch, z_vals, a_vals)

		batch_gradient = self.get_error(a_vals[-1], labels) * sigmoid_prime(z_vals[-1])

		bias_gradient_list = [0] * len(self.biases)
		weight_gradient_list = [0] * len(self.weights)

		bias_gradient_list[-1] = batch_gradient
		a = a_vals[-2]
		weight_gradient_list[-1] = np.matmul(batch_gradient, np.reshape(a, (a.shape[0], a.shape[2], a.shape[1])))

		a_vals_len = len(a_vals)
		for layer in range(2, self.no_of_layers):
			batch_gradient = np.matmul(self.weights[1 - layer].transpose(), batch_gradient) * sigmoid_prime(z_vals[-layer])
			bias_gradient_list[-layer] = batch_gradient

			a = a_vals[(-layer-1)%a_vals_len]
			weight_gradient_list[-layer] = np.matmul(batch_gradient, np.reshape(a, (a.shape[0], a.shape[2], a.shape[1])))

		weight_gradient_list = self.reduce_3_to_2_dim_tensor(weight_gradient_list)
		bias_gradient_list = self.reduce_3_to_2_dim_tensor(bias_gradient_list)

		#sgd
		self.weights = [w - ((learning_rate/len(mini_batch.shape)) * gw) for w, gw in zip(self.weights, weight_gradient_list)]
		self.biases = [b - ((learning_rate/len(mini_batch.shape)) * gb) for b, gb in zip(self.biases, bias_gradient_list)]
		
	def reduce_3_to_2_dim_tensor(self, tensor_list):
		gradient_list = []
		for gradient in tensor_list:
			gradient_sum = np.zeros((gradient.shape[1], gradient.shape[2]))
			for err in gradient:
				gradient_sum += err
			gradient_list.append(gradient_sum)
		return gradient_list

	def get_error(self, a_vals, labels):
		return (a_vals - labels)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	s = sigmoid(z)
	return s * (1 - s)

if __name__ == "__main__":
	import mnist_loader
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	net = Network([784, 30, 10])
	net.train(list(training_data), 30, 10, 3.0, validation_data=np.array(list(test_data)))