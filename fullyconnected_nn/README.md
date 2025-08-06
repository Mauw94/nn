````markdown
# 🧠 Binary Neural Network (Feedforward) for Image Classification

This project implements a **basic feedforward neural network** (fully connected) from scratch in NumPy, designed for **binary classification tasks**, such as distinguishing between **cats and dogs** in images.

---

## 📌 Overview

This neural network works by feeding raw image pixels through several dense (fully connected) layers, applying nonlinear activation functions like **ReLU** and **Sigmoid**, and training the model with **gradient descent** using **binary cross-entropy loss**.

---

## 🧩 Network Architecture

We define the network structure via a list of integers, where each value corresponds to the number of neurons in a layer.

Example:

```python
model = BinaryNeuralNet(layers=[64*64*3, 128, 64, 1])
````

This creates a network with:

* **Input layer**: `64*64*3 = 12288` neurons (e.g., 64x64 RGB image flattened a.k.a. 12288 pixels)
* **Hidden layers**: (hidden layers adjusted based on the training needs, we can add layers, or increase/decrease the amount of neurons)

  * 128 neurons
  * 64 neurons
* **Output layer**: 1 neuron (for binary classification)

---

## 📥 Input Representation

The network expects input images as flattened arrays of shape:

```
(batch_size, input_size)
```

For example, a batch of 100 RGB images of size 64x64 would be shaped like:

```
(100, 64*64*3)
```

Each row contains the pixel values of one image.

---

## 🧮 Weights & Biases

* **Weights (`self.weights`)**: List of matrices, where each matrix connects one layer to the next.
  Each matrix is initialized using He initialization:

  ```python
  np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
  ```

* **Biases (`self.biases`)**: List of vectors, each initialized to zero.
  Each bias vector corresponds to the neurons in the next layer.

---

## 🔁 Forward Propagation

The `forward(X)` method passes data through the network layers:

1. **Compute pre-activations (`z`)**:

   ```
   z = a @ W + b
   ```
2. **Apply activations**:

   * `ReLU` for hidden layers
   * `Sigmoid` for output layer (since it's a binary classifier)

Example forward pass:

```python
a = relu(np.dot(a, W1) + b1)
a = relu(np.dot(a, W2) + b2)
a = sigmoid(np.dot(a, W3) + b3)
```

Returns:

* `a[-1]`: Final predictions (probabilities)
* `z`: Pre-activation values for each layer (used in backprop)

---

## 🔁 Backward Propagation

The `backward(X, y)` method uses **binary cross-entropy loss** and performs gradient descent.

Steps:

1. **Calculate error at output**:

   ```python
   delta = a[-1] - y  # assuming sigmoid + BCE
   ```

2. **Iterate backwards through layers**:

   * Compute gradients:

     ```python
     grad_w = a[i].T @ delta
     grad_b = sum(delta)
     ```
   * Update weights and biases using:

     ```python
     W -= learning_rate * grad_w
     b -= learning_rate * grad_b
     ```
   * Propagate the error:

     ```python
     delta = (delta @ W.T) * relu_derivative(z[i-1])
     ```

---

## 🧠 Activation Functions

* **ReLU**: `f(x) = max(0, x)`
* **Sigmoid**: `f(x) = 1 / (1 + exp(-x))`

These are defined in `shared/activation.py`.

---

## 🎯 Prediction

* `predict_proba(X)` returns predicted probabilities between 0 and 1.
* `predict(X)` returns binary class predictions: 0 or 1.

---

## 💾 Save & Load

You can persist the trained model:

```python
model.save('model.npz')   # Saves weights and biases
model.load('model.npz')   # Restores the same state
```

Weights and biases are stored using NumPy's `.npz` format.

---

## 🧪 Training Loop

Training is typically done like this:

```python
for epoch in range(epochs):
    model.forward(X_train)
    model.backward(X_train, y_train, learning_rate=0.01)
```

Loss function: **Binary Cross-Entropy (BCE)**

```python
loss = -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred))
```

Accuracy: Percentage of correct predictions (based on a threshold of 0.5).

---

## 📊 Example Use Case

Binary classification of **Cats vs. Dogs** using images from the `PetImages/` directory. (PetImages not included in the repo due to size)

```python
model = BinaryNeuralNet(layers=[64*64*3, 128, 64, 1])
data, labels = load_data(...)
X_train, y_train, X_test, y_test = split_data(data, labels)
train(model, X_train, y_train, epochs=100)
```

---

## 🧠 Interface Compatibility

This model inherits from:

```python
class NeuralNetInterface:
    def forward(self, X): 
    def backward(self, X, y, learning_rate): .
    def predict_proba(self, X): 
    def predict(self, X): 
    def save(self, path): 
    def load(self, path): 
```

Which makes it easy to plug into existing training, evaluation, or inference pipelines.

---

## 🛠 Future Improvements

* Add regularization (e.g., L2 or dropout)
* Add learning rate decay or momentum
* Support multi-class classification
* Switch to softmax output for multi-label classification

---

## 📂 File Structure

```
📦binary
 ┣ 📜nn.py               ← BinaryNeuralNet class
 ┣ 📜activation.py       ← ReLU, sigmoid, etc.
 ┣ 📜train.py            ← Training functions
 ┣ 📜utils.py            ← Data loading, preprocessing
 ┗ 📜main.py             ← Entry point
```

```