import numpy as np

# Neural Network Model for Image Classification
# Forward: passes data through the network to get predictions.
# Backward: updates weights and biases based on the error of predictions.
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
        # 3. Output layer with 2 neurons (for the two classes)
        self.layers = layers
        self.weights = [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.random.rand(1, layers[i+1]) for i in range(len(layers)-1)]

        print("Neural Network initialized with layers:", self.layers)
 
    def sigmoid_activation(self, x):
            z = np.clip(z, -500, 500)  # clip values to prevent overflow
            return 1 / (1 + np.exp(-z))
    
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
    
    def tahn_derivative(self, x):
        t = self.tanh(x)
        return 1 - t**2
    
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    # Computes the output of the network for input X
    # by passing it through the layers
    def forward(self, X):
        a = [X]
        z = []
        for i in range(len(self.layers)-1):
            zi = np.dot(a[i], self.weights[i]) + self.biases[i]
            ai = self.sigmoid_activation(zi)
            z.append(zi)
            a.append(ai)
        return a, z
    
    # Updates the weights and biases using backpropagation using gradient descent
    # based on the error between predicted and actual output
    def backward(self, X, y, learning_rate=0.01):
         a, z = self.forward(X)
         m = X.shape[0]
         delta = a[-1] - y  # assumes sigmoid + BCE
         for i in reversed(range(len(self.layers) - 1)):
            grad_w = np.dot(a[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z[i - 1])
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.a[-1]))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        a, _ = self.forward(X)
        return (a[-1] > 0.5).astype(int)
    
    def save(self, path):
        np.savez(path, weights=self.weights, biases=self.biases)
    
    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        print(f"Model loaded from {path} with weights and biases.")


# When to use which activation function:
# | Function   | Formula                                    | Derivative                               | Usage                    |
# | ---------- | ------------------------------------------ | ---------------------------------------- | ------------------------ |
# | Sigmoid    | $\frac{1}{1 + e^{-x}}$                     | $\sigma(x)(1 - \sigma(x))$               | Binary classification    |
# | ReLU       | $\max(0, x)$                               | $1 \text{ if } x > 0 \text{ else } 0$    | Hidden layers            |
# | Leaky ReLU | $x \text{ if } x>0 \text{ else } \alpha x$ | $1 \text{ if } x>0 \text{ else } \alpha$ | Hidden layers            |
# | Tanh       | $\tanh(x)$                                 | $1 - \tanh^2(x)$                         | Hidden layers (older NN) |
# | Softmax    | $\frac{e^{x_i}}{\sum e^{x_j}}$             | Complex (use `y_pred - y_true`)          | Output for multi-class   |
