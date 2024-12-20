# Neural Network C++ Implementation

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project is a **Multi-Layer Perceptron (MLP)** neural network implemented in C++ using the [Eigen](https://eigen.tuxfamily.org/) library for  matrix operations. The neural network is designed to perform classification tasks, with flexibility in configuring the number of layers, hidden units, learning rate, and training iterations. 

## Features

- **Configurable Architecture:** Specify the number of layers, hidden units, and output classes.
- **Activation Functions:** Utilizes ReLU for hidden layers and Softmax for the output layer.
- **Training:** Implements forward and backward propagation with gradient descent optimization.
- **Accuracy Calculation:** Evaluates the model's performance on training data.
- **Utility Functions:** Includes functions for one-hot encoding and activation computations.
- **Command-Line Interface:** Interactive prompts to input training parameters and dataset.

## Demo

![Neural Network Training](https://github.com/DanTheCoderMan06/neural-network-cpp/blob/main/demo/training.gif)

*Figure: Training progress showing iteration counts and accuracy metrics.*

## Installation

### Prerequisites

- **C++ Compiler:** Supports C++17 standard.
- **CMake:** Version 3.10 or higher.
- **Eigen Library:** Installed on your system.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DanTheCoderMan06/neural-network-cpp.git
   cd neural-network-cpp
   ```

2. **Install Eigen**

   - **Ubuntu:**

     ```bash
     sudo apt-get update
     sudo apt-get install libeigen3-dev
     ```

   - **MacOS (using Homebrew):**

     ```bash
     brew install eigen
     ```

   - **Windows:**

     - Download Eigen from [Eigen Downloads](https://gitlab.com/libeigen/eigen/-/releases).
     - Extract and place it in a known directory.

3. **Build the Project**

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

   *Note: If Eigen is installed in a non-standard directory, you might need to specify the path during the CMake configuration.*

## Usage

After building the project, you can run the executable to train the neural network on your dataset.

```bash
./rooster
```

### Input Prompts

1. **Dataset Filename:**
   
   - Provide the path to your training CSV file (e.g., `train.csv`).
   - **CSV Format:** Each row should start with a label followed by pixel values (e.g., for MNIST, labels `0-9` followed by `784` pixel values).

2. **Number of Iterations:**
   
   - Specify how many training iterations to perform.
   - Must be a multiple of 25. If not, it will be rounded down to the nearest multiple.

3. **Number of Layers:**
   
   - Define the number of layers in the Multi-Layer Perceptron.
   - Minimum of 2 layers (1 hidden layer + 1 output layer).

4. **Learning Rate:**
   
   - Set the learning rate (e.g., `0.1`).

### Example Interaction

```
Enter filename (e.g., train.csv): train.csv
How many iterations shall happen? (Must be multiple of 25, if not is rounded): 500
How many layers in the Multi-Layer Perceptron? 3
Learning rate? 0.1
Starting training for 500 iterations...
Iteration: 0
Accuracy: 0.12
Iteration: 20
Accuracy: 0.45
...
Iteration: 500
Accuracy: 0.85
Training completed.
Final training accuracy: 0.85
```

## Project Structure

```
neural-network-cpp/
├── CMakeLists.txt          # CMake configuration file
├── README.md               # Project documentation
├── main.cpp                # Entry point of the application
├── NeuralNetwork.hpp       # NeuralNetwork class declaration
├── NeuralNetwork.cpp       # NeuralNetwork class implementation
├── utils.hpp               # Utility function declarations
├── utils.cpp               # Utility function implementations
├── data/                   # Directory for datasets
│   └── train.csv
└── build/                  # Directory for build files
```

## Dependencies

- **[Eigen](https://eigen.tuxfamily.org/):** A high-performance C++ library for linear algebra.
- **C++17 Standard:** Ensure your compiler supports C++17.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

- **Author:** Daniil Novak
- **Email:** daniilnovak@ucsb.edu
- **GitHub:** [@DanTheCoderMan06](https://github.com/DanTheCoderMan06)

Feel free to reach out for any questions or suggestions!

## Acknowledgments

- Inspired by various neural network implementations and educational resources.
- Thanks to the [Eigen](https://eigen.tuxfamily.org/) community for providing a robust linear algebra library.
