from binary.src.model import NeuralNetwork
from binary.src.utils import load_data, split_data, one_hot_encode
from binary.src.train import train, evaluate, predict_random_image
import argparse

def main(args):    
    model = NeuralNetwork(layers=[64*64*3, 128, 1]) # 1 output neuron for binary classification (Cat vs Dog)

    if args.load_model:
        model.load(args.load_model)
        print(f"Model loaded from {args.load_model}")
        if args.predict_random_image:
            predict_random_image(model, 'PetImages', ['Cat', 'Dog'], image_size=(64, 64))
            return

    data, labels = load_data('PetImages', ['Cat', 'Dog'], max_images=100, image_size=(64, 64), normalize_image=True)
    
    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(data, labels, seed=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # y_train = one_hot_encode(y_train, num_classes=2)
    # y_test = one_hot_encode(y_test, num_classes=2) => one-hot is for multi-class, not binary

    if not args.load_model:
        train(model, X_train, y_train, epochs=1000, learning_rate=0.01)
        if args.save_model:
            model.save(args.save_model)\

    evaluate(model, X_test, y_test)
    print("\n\n ----------------------------------- \n\n")
    for _ in range(10):
        predict_random_image(model, 'PetImages', ['Cat', 'Dog'], image_size=(64, 64))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Neural Network for Image Classification')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--save-model', type=str, default='model.npz', help='Path to save the trained model')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--predict-random-image', action='store_true', help='Predict a random image from the dataset')
    args = parser.parse_args()
    main(args)