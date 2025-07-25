# nn-image-classification

This project implements a neural network from scratch for image classification tasks. The neural network is designed to learn from image data and make predictions based on the learned patterns.

## Project Structure

```
binary/
├── src/
│   ├── main.py          # Entry point of the application
│   ├── model.py         # Defines the neural network architecture
│   ├── train.py         # Contains functions for training the neural network
│   ├── test.py          # Includes functions for testing the neural network's performance
│   └── utils.py         # Provides utility functions for data handling and preprocessing
├── requirements.txt      # Lists the dependencies required for the project
└── data/
    └── README.md        # Information about the dataset used
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nn
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

This will initialize the program and may call functions to train or test the neural network.

## Overview

The neural network implementation includes:
- **Model Architecture**: Defined in `model.py`, where layers and activation functions are constructed.
- **Training Loop**: Implemented in `train.py`, which handles the optimization of weights and loss calculation.
- **Testing Functions**: Found in `test.py`, which evaluates the model's performance on validation datasets.
- **Utility Functions**: Located in `utils.py`, these functions assist with data preprocessing and loading.

## Dataset

Refer to `data/README.md` for details on the dataset used for training and testing, including structure and preprocessing steps.