import os
from PIL import Image
import numpy as np

def load_data(image_dir, labels, image_size=(64, 64), max_images=None, normalize_image=True):
    X = []
    y = []
    i = 0

    for label, folder in enumerate(labels):
        folder_path = os.path.join(image_dir, folder)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    img = Image.open(file_path).convert('RGB')
                img = img.resize(image_size)
                X.append(normalize(img) if normalize_image else np.array(img).flatten())
                y.append(label)

                i += 1
                if i % 1000 == 0:
                    print(f"Loaded {i} images...")
                if max_images and i > max_images:
                    break
       
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
                continue
    
    X = np.array(X)
    y = np.array(y)
    return X, y

def normalize(img):
    """Normalizes the pixel values of an image to the range [0, 1].
    """
    return np.array(img).flatten() / 255.0

def preprocess_data(data):
    """
    Preprocess the data by normalizing pixel values. (Can be used after loading the data when normalize_image is False)
    """
    return data / 255.0 if isinstance(data, np.ndarray) else np.array(data) / 255.0

def split_data(data, labels, train_size=0.8, seed=None):
    """
    Split the data into training and testing sets.
    """
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data_shuffled = data[indices]
    labels_shuffled = labels[indices]
    split_index = int(len(data) * train_size)
    return (
        data_shuffled[:split_index], labels_shuffled[:split_index],
        data_shuffled[split_index:], labels_shuffled[split_index:]
    )

def one_hot_encode(labels, num_classes):
    """
    Convert labels to one-hot encoded format.
    """
    one_hot = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    return one_hot
