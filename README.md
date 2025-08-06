# General Overview of Neural Networks and Training Pipeline

## What is a Neural Network?

A **Neural Network (NN)** is a computational model inspired by the brain’s neurons, made of layers of nodes (neurons) that process data through learned weights and biases.

### Basic Components:

* **Input Layer:** Receives raw data (e.g., pixels of an image).
* **Hidden Layers:** Learn features by applying weights, biases, and nonlinear activations (e.g., ReLU).
* **Output Layer:** Outputs predictions (e.g., probability for binary classification).

---

## Forward Pass (Inference)

Data flows from input through hidden layers to output:

```python
def forward(self, X):
    a = X  # Input data
    self.a = [a]  # Store activations
    for i in range(len(self.weights)):
        z = np.dot(a, self.weights[i]) + self.biases[i]  # Linear transformation
        a = relu(z) if i < len(self.weights) - 1 else sigmoid(z)  # Activation
        self.a.append(a)
    return a  # Final output (prediction)
```

---

## Backward Pass (Training / Backpropagation)

Update weights/biases based on error between prediction and true labels:

```python
def backward(self, X, y, learning_rate):
    m = X.shape[0]
    delta = self.a[-1] - y  # Error at output
    for i in reversed(range(len(self.weights))):
        grad_w = np.dot(self.a[i].T, delta) / m  # Gradient wrt weights
        grad_b = np.sum(delta, axis=0, keepdims=True) / m  # Gradient wrt biases
        self.weights[i] -= learning_rate * grad_w  # Update weights
        self.biases[i] -= learning_rate * grad_b  # Update biases
        if i != 0:
            delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.a[i])  # Propagate error backward
```

---

## Data Loading and Splitting

Load data, normalize, and split into training/testing:

```python
def split_data(data, labels, train_ratio=0.8):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split_idx = int(len(data) * train_ratio)
    return (
        data[indices[:split_idx]], labels[indices[:split_idx]],
        data[indices[split_idx:]], labels[indices[split_idx:]]
    )
```

---

## Training Loop

Train the model over multiple epochs:

```python
def train(model, X_train, y_train, epochs=100, learning_rate=0.01):
    for epoch in range(epochs):
        model.forward(X_train)
        model.backward(X_train, y_train, learning_rate)
        if epoch % 10 == 0:
            preds = model.a[-1]
            loss = binary_cross_entropy(y_train, preds)
            acc = accuracy(y_train, preds)
            print(f"Epoch {epoch} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")
```

---

## What is Convolution and Max Pooling?

* **Convolution:** Applies small filters (kernels) sliding over input to detect local patterns like edges or textures in images. It reduces parameters and preserves spatial information.

* **Max Pooling:** Downsamples the feature map by selecting the maximum value in patches (e.g., 2x2), reducing spatial size and computation, while keeping important features.

---

## Summary

Neural networks transform raw inputs through layers of learned parameters to make predictions. They train by minimizing prediction errors via forward and backward passes, using data split carefully to ensure model generalization. CNNs extend this by focusing on spatial features, crucial for images and videos.

---

## Extra Notes

---

### 🧩 What is Convolution?

Convolution is a fundamental operation in CNNs where small filters (also called kernels) slide over the input image or feature map to detect specific patterns like edges, textures, or shapes. Each filter produces a feature map by computing weighted sums of the input pixels it covers. These learned filters allow the network to automatically extract important visual features without manual engineering.

**Key points:**

* Filters are small (e.g., 3x3 or 5x5).
* Each filter detects a specific feature.
* Output is a set of feature maps representing different detected patterns.

---

### 🧩 What is Max Pooling?

Max pooling is a downsampling technique that reduces the spatial dimensions (height and width) of the feature maps, keeping only the most important information. It works by sliding a window (e.g., 2x2) over the feature map and taking the maximum value within each window. This helps:

* Reduce computational cost
* Make features more robust to small translations or distortions
* Reduce overfitting by providing a form of spatial invariance
