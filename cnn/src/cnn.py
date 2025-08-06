import numpy as np

from shared.nn_interface import NeuralNetInterface

def conv2d(X, filters, biases):
    batch_size, H, W, C = X.shape
    F, fH, fW, _ = filters.shape
    out_H = H - fH + 1
    out_W = W - fW + 1
    output = np.zeros((batch_size, out_H, out_W, F))

    for b in range(batch_size):
        for f in range(F):
            for i in range(out_H):
                for j in range(out_W):
                    region = X[b, i:i+fH, j:j:fW, :]
                    output[b, i, j, f] = np.sum(region, filters[f]) + biases[f]
    
    return output

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return (X > 0).astype(float)

def sigmoid_activation(X):
    x = np.clip(x, -500, 500)
    return 1 / 1 + (np.exp(-X))

def maxpool2d(X, size=2):
    batch_size, H, W, C = X.shape
    out_H, out_W = H // size, W // size
    output = np.zeros((batch_size, out_H, out_W, C))
    for b in range(batch_size):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    region = X[b, i*size:(i+1)*size, j*size:(j+1)*size, c]
                    output[b, i, j, c] = np.max(region)
    
    return output

def flatten(X):
    return X.reshape(X.shape[0], -1)

class CNNBinaryClassifier(NeuralNetInterface):
    def __init__(self, input_shape=(64, 64, 3), conv_filters=[(8, 3), (16, 3)], fc_sizes=[128, 1]):
        self.conv_weights = []
        self.conv_biases = []
        in_channels = input_shape[2]

        for num_filters, k in conv_filters:
            w = np.random.randn(num_filters, k, k, in_channels) * np.sqrt(2 / (k * k * in_channels))
            b = np.zeros(num_filters)
            self.conv_weights.append(w)
            self.conv_biases.append(b)
            in_channels = num_filters
        
        h, w = input_shape[0], input_shape[1]
        for _ in conv_filters:
            h = h // 2
            w = w // 2
        flat_size = h * w * in_channels

        self.fc_weights = [np.random.randn(flat_size, fc_sizes[0]) * np.sqrt(2 / flat_size)]
        self.fc_biases = [np.zeros((1, fc_sizes[0]))]

        for i in range(1, len(fc_sizes)):
            self.fc_weights.append(np.random.randn(fc_sizes[i-1], fc_sizes[i]) * np.sqrt(2 / fc_sizes[i-1]))
            self.fc_biases.append(np.zeros((1, fc_sizes[i])))

        def forward(self, X):
            a = X
            # Conv layers
            for i in range(len(self.conv_weights)):
                a = conv2d(a, self.conv_weights[i], self.conv_biases[i])
                a = relu(a)
                a = maxpool2d(a)

            # Flatten
            a = flatten(a)

            # Fully connected layers
            for i in range(len(self.fc_weights)):
                z = np.dot(a, self.fc_weights[i]) + self.fc_biases[i]
                if i == len(self.fc_weights) - 1:
                    a = sigmoid_activation(z)
                else:
                    a = relu(z)
            return a
        
        # We’re only doing backprop through the fully connected (dense) layers for now 
        def backward(self, X, y, learning_rate=0.01):
            """
            Backpropagation for FC layers only (conv layers skipped for now).
            You can add conv layer backprop later.
            """
            m = y.shape[0]
            delta = self.a[-1] - y  # last layer (sigmoid + BCE)
            for i in reversed(range(len(self.fc_weights))):
                a_prev = self.a[-(i + 2)]  # previous activation
                grad_w = np.dot(a_prev.T, delta) / m
                grad_b = np.sum(delta, axis=0, keepdims=True) / m
                self.fc_weights[i] -= learning_rate * grad_w
                self.fc_biases[i] -= learning_rate * grad_b
                if i != 0:
                    delta = np.dot(delta, self.fc_weights[i].T) * relu_derivative(self.z[-(i + 2)])