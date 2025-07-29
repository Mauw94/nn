import numpy as np
from shared.activation import sigmoid_activation, relu_derivative, relu

# Neural Network Model for Binary Classification
class BinaryNeuralNet:
    def __init__(self, layers):
        # Example layers: [64*64*3, 128, 1]
        # where 64*64*3 is the input layer size (for 64x64 RGB images (*3 for RGB channels)),
        # 128 is a hidden layer size, so the hidden layer has 128 neurons. 
        # (We can add more hidden layers by extending e.g [64*64*3, 128, 64, 1]) -> We now have 2 hidden layers with 128 and 64 neurons respectively.)
        # and 1 is the output layer size. This is a binary classifier, so the output can only be 0 or 1.
        # If we create a different nn where the output are different classes (e.g. Dog, Cat, Hippo, Elephant, ...) we increase the output layer size.
        # So this neural network has 3 layers:
        # 1. Input layer with 64*64*3 neurons (for each pixel in a 64x64 RGB image)
        # 2. Hidden layer with 128 neurons
        # 3. Output layer with 1 neuron
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i+1]))for i in range(len(layers)-1)]

        print("nn initialized with layers: ", self.layers)

    # Computes the output of the network for input X by passing it through the layers
    def forward(self, X):
        self.a = [X] # activations
        self.z = [] # pre-activations
        for i in range(len(self.layers)-1):
            zi = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.z.append(zi)
            if i == len(self.layers) - 2:  # last layer
                ai = sigmoid_activation(zi) # sigmoid for binary classification
            else:
                ai = relu(zi) # use ReLU for hidden layers
            self.a.append(ai)
        return self.a, self.z
    
    # Updates the weights and biases using backpropagation using gradient descent based on the error between predicted and actual output
    def backward(self, X, y, learning_rate=0.01):
         assert self.a[-1].shape == y.shape, f"Shape mismatch: y_pred={self.a[-1].shape}, y={y.shape}"
         m = X.shape[0]
         # For binary classification, we can use sigmoid activation and binary cross-entropy loss (see train.py)
         delta = self.a[-1] - y  # assumes sigmoid + BCE 
         for i in reversed(range(len(self.layers) - 1)):
            grad_w = np.dot(self.a[i].T, delta) / m # gradient of the loss with respect to the weights (how much to change each weight to reduce loss)
            grad_b = np.sum(delta, axis=0, keepdims=True) / m # gradient of the loss with respect to the biases (how much to change each bias to reduce loss)
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.z[i - 1])

    def predict_proba(self, X):
        a, _ = self.forward(X)
        return a[-1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def save(self, path):
        try:
            data = {}
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                data[f"W{i}"] = w
                data[f"b{i}"] = b
            np.savez(path, **data)
        except Exception as e:
            print(f"Couldn't save the model due to error: {e}")
    
    def load(self, path):
        data = np.load(path)
        print(data)
        num_layers = len(self.layers) - 1
        self.weights = [data[f"W{i}"] for i in range(num_layers)]
        self.biases  = [data[f"b{i}"] for i in range(num_layers)]
        print(f"Model loaded from {path} with weights and biases.")
