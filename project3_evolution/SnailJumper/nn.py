import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        input_layer_size, hidden_layer_size, output_layer_size = layer_sizes
        self.input_layer_size = input_layer_size
        r = 0.4
        self.wh = np.random.randn(input_layer_size, hidden_layer_size) * r
        self.bh = np.zeros((1, hidden_layer_size))
        self.wo = np.random.randn(hidden_layer_size, output_layer_size) * r
        self.bo = np.zeros((1, output_layer_size))

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return 1 / (1 + np.exp(-x))

    def cal(self, a, w, b):
        res = np.dot(a, w) + b
        return self.activation(res)

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        r = self.cal(x, self.wh, self.bh)
        r = self.cal(r, self.wo, self.bo)
        return r[0][0] > r[0][1]
