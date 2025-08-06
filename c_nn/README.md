````markdown
# Convolutional Neural Network for Binary Image Classification

This project implements a simple Convolutional Neural Network (CNN) from scratch using NumPy, intended for binary classification (e.g., Cats vs Dogs). The model is designed to work with RGB images of shape 64x64x3 and is trained using forward and backward propagation logic tailored to convolutional and fully connected layers.

---

## 📦 Architecture Overview

The CNN is built with the following structure:

### 🔷 Input
- Shape: `(batch_size, 64, 64, 3)` — 64x64 RGB image per sample

### 🔷 Convolutional Layers
Defined via:
```python
conv_filters = [(16, 3), (32, 3)]
````

This means:

* Layer 1: 16 filters of size 3x3
* Layer 2: 32 filters of size 3x3
  Each layer applies:

1. Convolution
2. ReLU activation
3. MaxPooling (2x2) for downsampling

### 🔷 Fully Connected (Dense) Layers

After flattening the output of the convolutional stack:

```python
fc_layers = [128, 64, 1]
```

This means:

* Dense layer with 128 neurons (ReLU)
* Dense layer with 64 neurons (ReLU)
* Output layer with 1 neuron (Sigmoid for binary classification)

---

## 🔁 Forward Pass

The `forward(X)` function performs the following:

1. **Convolution + ReLU + MaxPool** over each conv layer.
2. **Flattening** of the final pooled feature maps.
3. **Dense layer computation**:

   * Uses `ReLU` for hidden layers.
   * Uses `Sigmoid` for the output layer (for binary prediction).

---

## 🔁 Backward Pass

The `backward(X, y)` function uses:

* **Binary Cross Entropy (BCE)** for the loss function.
* **Gradient descent** to update:

  * Fully connected weights and biases
  * (Convolutional layer updates not included in current implementation — can be extended!)

Gradients are computed using:

* Chain rule (standard backpropagation)
* ReLU derivatives for hidden layers
* Sigmoid derivative + BCE gradient for the final output layer

---

## ⚙️ Training

You can train the CNN with:

```python
train(model, X_train, y_train, epochs=100, learning_rate=0.001)
```

Inputs:

* `X_train`: 4D array with shape `(num_samples, 64, 64, 3)`
* `y_train`: Labels reshaped to `(num_samples, 1)`

---

## 📈 Prediction

* `model.predict_proba(X)` — returns probability between 0 and 1
* `model.predict(X)` — returns 0 or 1 depending on threshold (0.5 by default)

---

## 💾 Saving & Loading

* `model.save("path.npz")`: Save weights and biases
* `model.load("path.npz")`: Load from file

This allows model reuse without retraining.

---

## 🧠 Notes

* **All filters are learnable**, initialized using He initialization (`np.sqrt(2 / fan_in)`).
* Assumes preprocessed (resized + normalized) images.
* Designed for binary classification — can be extended to multiclass with softmax + categorical cross-entropy.

---

## 🔧 To Do / Extend

* Implement backpropagation through convolutional layers
* Add dropout or batch normalization
* Add data augmentation for better generalization
* Extend to multiclass classification
* Add support for different activation functions

---

## 📚 Dependencies

* Python 3
* NumPy
* (Optional) PIL / OpenCV for image loading and resizing

---

## 🐾 Example Use Case

Training on `PetImages` folder with two subfolders `Cat/` and `Dog/`, resized to 64x64 RGB.
Model learns to distinguish between cats and dogs using pixel-level and mid-level features via convolutions.

---

```