import numpy as np
import os

from shared.nn_interface import BinaryNNInterface

def train(model: BinaryNNInterface, X, y, epochs=100, learning_rate=0.01):
    """
    Train the neural network model using the provided data.
    
    Parameters:
    - model: The NeuralNetwork instance to train.
    - X: Input data (features).
    - y: Target data (labels).
    - epochs: Number of training iterations.
    - learning_rate: Step size for weight updates.
    """
    for epoch in range(epochs):
        preds = model.forward(X)
        model.backward(X, y, learning_rate)
        if epoch % 1 == 0:
            acc = accuracy(y, preds)
            loss = binary_cross_entropy(y, preds)
            print(f'Epoch {epoch}, Loss: {loss}, Pred Mean: {preds.mean():.4f}, Min: {preds.min():.4f}, Max: {preds.max():.4f}, Acc: {acc}')
    
def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross-entropy loss.
    
    Parameters:
    - y_true: True labels (not hot encoded since this is a binary nn)
    - y_pred: Predicted probabilities from the model.
    
    Returns:
    - loss: The computed binary cross-entropy loss.
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true, y_pred, threshold=0.5):
    preds = (y_pred > threshold).astype(int)
    return np.mean(preds == y_true)

def evaluate(model: BinaryNNInterface, X, y):
    """
    Evaluate the model's performance on the provided data.
    
    Parameters:
    - model: The NeuralNetwork instance to evaluate.
    - X: Input data (features).
    - y: Target data (labels).
    
    Returns:
    - accuracy: The accuracy of the model on the provided data.
    """
    prediction = model.predict(X)
    print("First 10 predictions:", model.a[-1][:10].flatten())
    print("First 10 true labels     :", y[:10].flatten())
    print("Correct predictions      :", (prediction[:10].flatten() == y[:10].flatten()))
    acc = accuracy(y, model.a[-1])
    print(f'Accuracy: {acc * 100:.2f}%')
    return acc

def predict_random_image(model: BinaryNNInterface, image_dir, labels, image_size=(64, 64)):
    """
    Predict the class of a random image from the specified directory.
    
    Parameters:
    - model: The NeuralNetwork instance to use for prediction.
    - image_dir: Directory containing images.
    - labels: List of class labels.
    - image_size: Size to which images should be resized.
    
    Returns:
    - predicted_label: The predicted class label for the random image.
    """
    import random
    from PIL import Image
    
    folder = random.choice(labels)
    folder_path = os.path.join(image_dir, folder)
    filename = random.choice(os.listdir(folder_path))
    file_path = os.path.join(folder_path, filename)
    
    img = Image.open(file_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img).flatten() / 255.0  # Normalize
    img_array = img_array.reshape(1, -1)  # Reshape for prediction
    
    prediction = model.predict(img_array)  # shape (1, 1), value 0 or 1
    predicted_label = labels[int(prediction[0, 0])]

    print(f'Actual label: {folder}, Predicted label for {folder}{filename}: {predicted_label}')
    return predicted_label

def predict_image_from_path(model: BinaryNNInterface, file_path, labels, image_size=(64, 64)):
    from PIL import Image

    img = Image.open(file_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img).flatten() / 255.0  # Normalize
    img_array = img_array.reshape(1, -1)  # Reshape for prediction

    prediction = model.predict(img_array)
    predicted_label = labels[int(prediction[0][0])]
    print(f'Predicted label: {predicted_label}')
    return predicted_label
