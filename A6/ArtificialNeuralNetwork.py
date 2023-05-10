import pandas as pd
import numpy as np

class ArtificialNeuralNetwork:
    """
    The ArtificialNeuralNetwork class represents an implementation of an
    artificial neural network (ANN) for classification tasks.

    Methods:
    --------
        loadFile(file, col_names):
            Loads the data file as a pandas DataFrame and maps the class labels to integers.

        scale_data():
            Scales the data in the DataFrame.

        scale_row(row):
            Scales a given row of data.

        split(frac=0.8, random_state=1):
            Splits the data into training and validation sets.

        _sigmoid(x):
            Defines the sigmoid activation function.

        _sigmoid_derivative(x):
            Defines the derivative of the sigmoid activation function for backpropagation.

        train(train_data, num_epochs=1000, learning_rate=0.1, num_hidden=5):
            Trains the neural network on the training data.

        test(val_data):
            Tests the trained neural network on validation data and returns the accuracy.

        predict(row):
            Predicts the class of a given row of data.

    Attributes:
    -----------
        _data:
            Pandas DataFrame containing the loaded data.

        _min:
            Pandas Series containing the minimum values of each feature in the loaded data.

        _max:
            Pandas Series containing the maximum values of each feature in the loaded data.

        _scaled:
            Pandas DataFrame containing the scaled data.

        _class_to_int:
            Dictionary mapping class labels to integers.

        _weights_hidden:
            Numpy array containing the weights of the hidden layer.

        _bias_hidden:
            Numpy array containing the biases of the hidden layer.

        _weights_output:
            Numpy array containing the weights of the output layer.

        _bias_output:
            Numpy array containing the biases of the output layer.
    """

    def loadFile(self, file, col_names):
        """
        Loads the data file as a pandas DataFrame and maps the class labels to integers.
        Args:
            file (string): path to the data file to be loaded
            col_names (list): list of column names for the data file
        Returns:
            none
        """
        data = pd.read_csv(file, names=col_names)
        class_to_int = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        class_to_int = {key: int(value) for key, value in class_to_int.items()}
        data[data.columns[-1]] = data[data.columns[-1]].map(class_to_int)
        self._data = data
        self._min = self._data.iloc[:,:-1].min()
        self._max = self._data.iloc[:,:-1].max()
        self._class_to_int = class_to_int

    def scale_data(self):
        """
        Scales the input data by subtracting the minimum and dividing by the range.
        Args:
            none
        Returns:
            scaled_data (pandas DataFrame): the scaled input data
        """
        self._scaled = (self._data.iloc[:,:-1] - self._min) / (self._max - self._min)
        return self._scaled

    def scale_row(self, row):
        """
        Scales a single input row by subtracting the minimum and dividing by the range.
        Args:
            row (pandas Series): the input row to be scaled
        Returns:
            scaled_row (pandas Series): the scaled input row
        """
        return (row.iloc[:,:-1] - self._min) / (self._max - self._min)

    # def scale_row(self, row):
    def split(self, frac = 0.8, random_state = 1):
        """
        Splits the scaled data into a training set and a validation set.
        Args:
            frac (float): the fraction of data to use for training (default 0.8)
            random_state (int): the random seed to use for shuffling the data (default 1)
        Returns:
            train_data (pandas DataFrame): the training set
            val_data (pandas DataFrame): the validation set
        """
        train_data = self._scaled.sample(frac = 0.8, random_state=1)
        val_data = self._scaled.drop(train_data.index)
        return train_data, val_data

    # Define the activation function (e.g., sigmoid)
    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    # Define the derivative of the activation function for backpropagation
    def _sigmoid_derivative(self,x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def train(self, train_data, num_epochs = 1000, learning_rate = 0.1, num_hidden = 5):
        """
        Trains the neural network on the given training data.
        Args:
            train_data (pandas DataFrame): the training data
            num_epochs (int): the number of training epochs to perform (default 1000)
            learning_rate (float): the learning rate for gradient descent (default 0.1)
            num_hidden (int): the number of hidden units in the neural network (default 5)
        Returns:
            none
        """
        num_inputs = 4
        num_outputs = len(self._class_to_int)

        weights_hidden = np.random.randn(num_inputs,num_hidden)
        bias_hidden = np.random.randn(num_hidden)
        weights_output = np.random.randn(num_hidden, num_outputs)
        bias_output = np.random.randn(num_outputs)

        for epoch in range(num_epochs):
            for i, row in train_data.iterrows():
                # Forward pass
                hidden_layer_activation = np.dot(row, weights_hidden) + bias_hidden
                hidden_layer_output = self._sigmoid(hidden_layer_activation)
                output_layer_activation = np.dot(hidden_layer_output, weights_output) + bias_output
                predicted_output = self._sigmoid(output_layer_activation)

                # Calculate the error and delta for the output layer
                actual_output = np.zeros(num_outputs)
                actual_output[int(self._data.iloc[i,-1])] = 1 #hereeeeeeeeeeeeeeeeee
                error_output = predicted_output - actual_output
                delta_output = error_output * self._sigmoid_derivative(output_layer_activation)

                # Calculate the error and delta for the hidden layer
                error_hidden = np.dot(delta_output, weights_output.T)
                delta_hidden = error_hidden * self._sigmoid_derivative(hidden_layer_activation)

                # Update the weights and biases using the deltas
                weights_output -= learning_rate * np.outer(hidden_layer_output, delta_output)
                bias_output -= learning_rate * delta_output
                weights_hidden -= learning_rate * np.outer(row, delta_hidden)
                bias_hidden -= learning_rate * delta_hidden
                self._weights_hidden = weights_hidden
                self._bias_hidden = bias_hidden
                self._weights_output = weights_output
                self._bias_output = bias_output
    def test(self,val_data):
        """
        Tests the neural network on the given validation data and returns the accuracy.
        Args:
            val_data (pandas DataFrame): the validation data
        Returns:
            accuracy (float): the fraction of correctly classified examples in the validation data
        """
        num_correct = 0
        for i, row in val_data.iterrows():
            hidden_layer_activation = np.dot(row, self._weights_hidden) + self._bias_hidden
            hidden_layer_output = self._sigmoid(hidden_layer_activation)
            output_layer_activation = np.dot(hidden_layer_output, self._weights_output) + self._bias_output
            predicted_output = self._sigmoid(output_layer_activation)
            predicted_class = np.argmax(predicted_output)
            actual_class = self._data.iloc[i,-1]
            if predicted_class == actual_class:
                num_correct += 1
        return num_correct/len(val_data)

    def predict(self, row):
        """
        Predicts the class label for a single input row.
        Args:
            row (pandas Series): the input row to be classified
        Returns:
            predicted_class (int): the predicted class label (0, 1, or 2)
        """
        hidden_layer_activation = np.dot(row, self._weights_hidden) + self._bias_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output, self._weights_output) + self._bias_output
        predicted_output = self._sigmoid(output_layer_activation)
        predicted_class = np.argmax(predicted_output)
        return predicted_class