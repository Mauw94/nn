import numpy as np
from cnn.src.utils import conv2d, flatten, maxpool2d
from shared.activation import sigmoid_activation, relu_derivative, relu
from shared.nn_interface import BinaryNNInterface


class CNNBinaryClassifier(BinaryNNInterface):
    def __init__(self, input_shape=(64, 64, 3), conv_filters=[(8, 3), (16, 3)], fc_sizes=[128, 1]):
        self.conv_weights = []
        self.conv_biases = []
        in_channels = input_shape[2]
        self.x = [] # activations (store for loss + backprop)
        self.z = [] # pre-activations

        for num_filters, k in conv_filters:
            w = np.random.randn(num_filters, k, k, in_channels) * np.sqrt(2 / (k * k * in_channels))
            b = np.zeros((num_filters,))
            self.conv_weights.append(w)
            self.conv_biases.append(b)
            in_channels = num_filters

        dummy_input = np.zeros((1, *input_shape))  # shape: (1, H, W, C)
        a = dummy_input
        for i in range(len(self.conv_weights)):
            z = conv2d(a, self.conv_weights[i], self.conv_biases[i])
            a = relu(z)
            a = maxpool2d(a)
        flat_size = flatten(a).shape[1]

        # fully connected layers
        self.fc_weights = [np.random.randn(flat_size, fc_sizes[0]) * np.sqrt(2 / flat_size)]
        self.fc_biases = [np.zeros((1, fc_sizes[0]))]

        for i in range(1, len(fc_sizes)):
            self.fc_weights.append(np.random.randn(fc_sizes[i - 1], fc_sizes[i]) * np.sqrt(2 / fc_sizes[i - 1]))
            self.fc_biases.append(np.zeros((1, fc_sizes[i])))

    def forward(self, X):
        # === Convolutional Layers ===
        a = X
        self.a = [a]  # store activations fom conv layers
        self.z = []   # store pre-activations from conv layers
        
        for i in range(len(self.conv_weights)):
            z = conv2d(a, self.conv_weights[i], self.conv_biases[i])
            a = relu(z)
            a = maxpool2d(a)
            self.z.append(z)
            self.a.append(a)
        
        a = flatten(a)
        self.z.append(a)  # just to match structure
        self.a.append(a)
        self.flattened_input = a # store flattened input for backdrop

         # === Fully Connected Layers ===
        self.fc_a = [a]  # first is the flattened input
        self.fc_z = []
        for i in range(len(self.fc_weights)):
            z = np.dot(a, self.fc_weights[i]) + self.fc_biases[i]
            a = sigmoid_activation(z) if i == len(self.fc_weights) - 1 else relu(z)
            self.fc_z.append(z)
            self.fc_a.append(a)

        return self.a[-1]

    
    # We’re only doing backprop through the fully connected (dense) layers for now 
    def backward(self, X, y, learning_rate=0.01):
        m = y.shape[0]

        # === Fully Connected Backprop ===
        delta = self.fc_a[-1] - y  # sigmoid + BCE derivative

        for i in reversed(range(len(self.fc_weights))):
            a_prev = self.fc_a[i]
            z = self.fc_z[i]

            grad_w = np.dot(a_prev.T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            self.fc_weights[i] -= learning_rate * grad_w
            self.fc_biases[i] -= learning_rate * grad_b

            if i > 0:
                delta = np.dot(delta, self.fc_weights[i].T) * relu_derivative(self.fc_z[i - 1])

        # no conv layer backdrop yet

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)