import numpy as np

# Neural Network Model for Image Classification
class NeuralNetwork:
    def __init__(self, layers):
        # Example layers: [64*64*3, 128, 2]
        # where 64*64*3 is the input layer size (for 64x64 RGB images (*3 for RGB channels)),
        # 128 is a hidden layer size, so the hidden layer has 128 neurons,
        # and 2 is the output layer size (for 2 classes).
        # e.g. Classes could be 'Cat' and 'Dog'. Hence, the output layer has 2 neurons.
        # So this neural network has 3 layers:
        # 1. Input layer with 64*64*3 neurons (for each pixel in a 64x64 RGB image)
        # 2. Hidden layer with 128 neurons
        # 3. Output layer with 1 neuron
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
        self.biases = [np.zeros((1, layers[i+1]))for i in range(len(layers)-1)]

        print("Neural Network initialized with layers: ", self.layers)
 
    # TODO: move activation/derivative functions to a separate file and use based on input
    def sigmoid_activation(self, x):
            x = np.clip(x, -500, 500)  # clip values to prevent overflow
            return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, X):
        s = self.sigmoid_activation(X)
        return s * (1 - s)

    def relu(self, X):
        return np.maximum(0, X)

    def relu_derivative(self, X):
        return (X > 0).astype(float)

    # Good for avoiding 'dead' neurons in ReLU activations
    def leaky_relu(self, X, alpha=0.01):
        return np.where(X > 0, X, alpha * X)              

    def leaky_relu_derivative(self, X, alpha=0.01):
        return np.where(X > 0, 1, alpha)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        t = self.tanh(x)
        return 1 - t**2
    
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    # Computes the output of the network for input X
    # by passing it through the layers
    def forward(self, X):
        self.a = [X] # activations
        self.z = [] # pre-activations
        for i in range(len(self.layers)-1):
            zi = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            self.z.append(zi)
            # ai = self.sigmoid_activation(zi)
            if i == len(self.layers) - 2:  # last layer
                ai = self.sigmoid_activation(zi) # sigmoid for binary classification
            else:
                ai = self.relu(zi) # use ReLU for hidden layers
            self.a.append(ai)
        return self.a, self.z
    
    # Updates the weights and biases using backpropagation using gradient descent
    # based on the error between predicted and actual output
    def backward(self, X, y, learning_rate=0.01):
         m = X.shape[0]
         assert self.a[-1].shape == y.shape, f"Shape mismatch: y_pred={self.a[-1].shape}, y={y.shape}"

         # For binary classification, we can use sigmoid activation and binary cross-entropy loss
         delta = self.a[-1] - y  # assumes sigmoid + BCE 
         for i in reversed(range(len(self.layers) - 1)):
            grad_w = np.dot(self.a[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z[i - 1])

    def predict_proba(self, X):
        a, _ = self.forward(X)
        return a[-1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def save(self, path):
        # Ensure all biases are 2D arrays with shape (1, N)
        biases_fixed = [b.reshape(1, -1) if b.ndim == 1 else b for b in self.biases]    
        # a = np.random.rand(1, 128)
        # b = np.random.rand(1, 2)
        # biases = [a, b]
        # TODO: still gives ValueError: could not broadcast input array from shape (128,) into shape (1,)
        try:
            np.savez(path, weights=np.array(self.weights, dtype=object), biases=np.array(biases_fixed, dtype=object))
        except Exception as e:
            print(f"Couldn't save the model due to error: {e}")
    
    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = [b.reshape(1, -1) if b.ndim == 1 else b for b in list(data['biases'])]
        print(f"Model loaded from {path} with weights and biases.")


# When to use which activation function:
# | Function   | Formula                                    | Derivative                               | Usage                    |
# | ---------- | ------------------------------------------ | ---------------------------------------- | ------------------------ |
# | Sigmoid    | $\frac{1}{1 + e^{-x}}$                     | $\sigma(x)(1 - \sigma(x))$               | Binary classification    |
# | ReLU       | $\max(0, x)$                               | $1 \text{ if } x > 0 \text{ else } 0$    | Hidden layers            |
# | Leaky ReLU | $x \text{ if } x>0 \text{ else } \alpha x$ | $1 \text{ if } x>0 \text{ else } \alpha$ | Hidden layers            |
# | Tanh       | $\tanh(x)$                                 | $1 - \tanh^2(x)$                         | Hidden layers (older NN) |
# | Softmax    | $\frac{e^{x_i}}{\sum e^{x_j}}$             | Complex (use `y_pred - y_true`)          | Output for multi-class   |
