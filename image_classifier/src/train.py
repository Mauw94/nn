from image_classifier.src.model import NeuralNetwork
import numpy as np
import os

def train(model: NeuralNetwork, X, y, epochs=100, learning_rate=0.01):
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
        model.forward(X)
        model.backward(X, y, learning_rate)
        # if epoch % 100 == 0:
        loss = np.mean(np.square(y - model.a[-1]))
        print(f'Epoch {epoch}, Loss: {loss}')
    

def evaluate(model: NeuralNetwork, X, y):
    """
    Evaluate the model's performance on the provided data.
    
    Parameters:
    - model: The NeuralNetwork instance to evaluate.
    - X: Input data (features).
    - y: Target data (labels).
    
    Returns:
    - accuracy: The accuracy of the model on the provided data.
    """
    predictions = model.predict(X)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y, axis=1)
    accuracy = np.mean(predicted_classes == true_classes)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def predict_random_image(model: NeuralNetwork, image_dir, labels, image_size=(64, 64)):
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
    
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    
    print(f'Predicted label for {filename}: {predicted_label}')
    return predicted_label