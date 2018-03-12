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
			d = np.dot(w, mini_batch)
			if z_vals != None:
				d = np.reshape(d, (d.shape[1], d.shape[0], d.shape[2]))
			z = d + b
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

		avg_gradient = self.reduce_3_to_2_dim_tensor(batch_gradient)

		bias_gradient_list = [np.zeros(b.shape) for b in self.biases]
		weight_gradient_list = [np.zeros(w.shape) for w in self.weights]
		bias_gradient_list[-1] = avg_gradient
		a = self.reduce_3_to_2_dim_tensor(a_vals[-2])

		weight_gradient_list[-1] = np.dot(avg_gradient, a.transpose())
		a_vals_len = len(a_vals)
		for layer in range(2, self.no_of_layers):
			avg_gradient = np.dot(self.weights[1 - layer].transpose(), avg_gradient) * sigmoid_prime(self.reduce_3_to_2_dim_tensor(z_vals[-layer]))
			bias_gradient_list[-layer] = avg_gradient

			a = self.reduce_3_to_2_dim_tensor(a_vals[(-layer-1)%a_vals_len])
			weight_gradient_list[-layer] = np.dot(avg_gradient, a.transpose())

		#sgd
		self.weights = [w - ((learning_rate/len(mini_batch.shape)) * gw) for w, gw in zip(self.weights, weight_gradient_list)]
		self.biases = [b - ((learning_rate/len(mini_batch.shape)) * gb) for b, gb in zip(self.biases, bias_gradient_list)]
		
	def reduce_3_to_2_dim_tensor(self, tensor):
		avg_gradient = np.zeros((tensor.shape[1], tensor.shape[2]))
		for err in tensor:
			avg_gradient += err

		avg_gradient /= tensor.shape[0]
		return avg_gradient

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
	net.train(list(training_data), 10, 10, 0.3, validation_data=np.array(list(test_data)))