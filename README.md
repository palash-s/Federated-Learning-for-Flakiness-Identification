# Federated Learning for Flaky Test Classification

This repository contains the implementation of a federated learning approach for classifying flaky and non-flaky tests using a neural network model. The script `eff-eff-FL.py` simulates federated learning by splitting the data across multiple clients and aggregating their model updates.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [Input and Output](#input-and-output)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The script `eff-eff-FL.py` implements a federated learning approach for training a neural network model to classify flaky and non-flaky tests. The code uses TensorFlow and Keras for building and training the neural network, and it simulates federated learning by splitting the data across multiple clients and aggregating their model updates.

## Features

- Federated learning simulation with multiple clients
- Neural network model for binary classification
- Data preprocessing and vectorization
- Model evaluation and metrics calculation
- Results saved to a CSV file

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/federated-learning-flaky-tests.git
    cd federated-learning-flaky-tests
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and place it in the `dataset` directory.

2. Run the script:
    ```bash
    python eff-eff-FL.py
    ```

3. The results will be saved to the `results/eff-eff-FEDL.csv` file.

## Functions

### `build_neural_network(input_shape)`

- **Description**: Builds and compiles a neural network model.
- **Input**: `input_shape` (tuple) - The shape of the input data.
- **Output**: A compiled Keras model.

### `preprocess_data(dataPointsList, dataLabelsList)`

- **Description**: Flattens the data points and converts labels to a numpy array.
- **Input**: `dataPointsList` (list of sparse matrices), `dataLabelsList` (list of labels).
- **Output**: `dataPointsList` (numpy array), `dataLabelsList` (numpy array).

### `create_tf_dataset(dataPointsList, dataLabelsList)`

- **Description**: Creates a TensorFlow dataset from the data points and labels.
- **Input**: `dataPointsList` (numpy array), `dataLabelsList` (numpy array).
- **Output**: A TensorFlow dataset.

### `sum_scaled_weights(scaled_weight_list)`

- **Description**: Aggregates the weights from multiple clients by summing them.
- **Input**: `scaled_weight_list` (list of weight lists).
- **Output**: `avg_grad` (list of aggregated weights).

### `test_model(X_test, Y_test, model, comm_round)`

- **Description**: Evaluates the model on the test data and prints the accuracy and loss.
- **Input**: `X_test` (numpy array), `Y_test` (numpy array), `model` (Keras model), `comm_round` (int).
- **Output**: `acc` (float), `loss` (float).

### `predict(model, data)`

- **Description**: Makes predictions using the model.
- **Input**: `model` (Keras model), `data` (TensorFlow dataset).
- **Output**: Predictions (TensorFlow tensor).

### `flastFederatedLearning(outDir, projectBasePath, projectName, kf, dim, eps)`

- **Description**: Implements the federated learning process.
- **Input**: `outDir` (str), `projectBasePath` (str), `projectName` (str), `kf` (StratifiedShuffleSplit), `dim` (int), `eps` (float).
- **Output**: A tuple containing various metrics.

## Input and Output

- **Input**:
  - `projectBasePath`: The base path to the dataset.
  - `projectList`: A list of project names to process.
  - `kf`: A StratifiedShuffleSplit object for cross-validation.
  - `dim`: The number of dimensions for vectorization.
  - `eps`: The epsilon value for vectorization.

- **Output**:
  - The results are written to a CSV file (`results/eff-eff-FEDL.csv`) containing the following columns:
    - `dataset`: The name of the dataset.
    - `flakyTrain`: The number of flaky tests in the training set.
    - `nonFlakyTrain`: The number of non-flaky tests in the training set.
    - `flakyTest`: The number of flaky tests in the test set.
    - `nonFlakyTest`: The number of non-flaky tests in the test set.
    - `precision`: The precision of the model.
    - `recall`: The recall of the model.
    - `storage`: The storage size of the model.
    - `preparationTime`: The time taken to prepare the data.
    - `predictionTime`: The time taken to make predictions.
    - `predictedFlaky`: The number of predicted flaky tests.
    - `predictedNonFlaky`: The number of predicted non-flaky tests.

## Results

The results of the federated learning process are saved to the `results/eff-eff-FEDL.csv` file. The file contains various metrics for each project, including precision, recall, preparation time, prediction time, and storage size.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.