import numpy as np

def sigmoid_activation(x):
        x = np.clip(x, -500, 500)  # clip values to prevent overflow
        return 1 / (1 + np.exp(-x))

def sigmoid_derivative(X):
    s = sigmoid_activation(X)
    return s * (1 - s)

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return (X > 0).astype(float)

# Good for avoiding 'dead' neurons in ReLU activations
def leaky_relu(X, alpha=0.01):
    return np.where(X > 0, X, alpha * X)              

def leaky_relu_derivative(X, alpha=0.01):
    return np.where(X > 0, 1, alpha)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2


# When to use which activation function:
# | Function   | Formula                                    | Derivative                               | Usage                    |
# | ---------- | ------------------------------------------ | ---------------------------------------- | ------------------------ |
# | Sigmoid    | $\frac{1}{1 + e^{-x}}$                     | $\sigma(x)(1 - \sigma(x))$               | Binary classification    |
# | ReLU       | $\max(0, x)$                               | $1 \text{ if } x > 0 \text{ else } 0$    | Hidden layers            |
# | Leaky ReLU | $x \text{ if } x>0 \text{ else } \alpha x$ | $1 \text{ if } x>0 \text{ else } \alpha$ | Hidden layers            |
# | Tanh       | $\tanh(x)$                                 | $1 - \tanh^2(x)$                         | Hidden layers (older NN) |
# | Softmax    | $\frac{e^{x_i}}{\sum e^{x_j}}$             | Complex (use `y_pred - y_true`)          | Output for multi-class   |
