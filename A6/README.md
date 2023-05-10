30 Apr 2023
Developed by Abdulrahman (Labib) Afia
Fabrizio Santini
A6 - ArtificialNeuralNetwork
===========================================================
Run command:
    python3 A6.py

Packages that need to be installed on your python version:
    Pandas
    Numpy

--------------------------------------------
I use the ArtificialNeuralNetwork class I created to do all the heavy lifting.
Find below the description, usage and underlying assumptions to make the most
out of it:

ArtificialNeuralNetwork Class:


Description:

This class provides an implementation of a simple artificial neural network with a single hidden layer for multiclass classification tasks. It is assumed that the data is in a CSV file format, and the class labels are in string format. The class labels are mapped to integers for use in the network.

Assumptions:

- Data is in a CSV file format
- Class labels are in string format
- All features are numerical and continuous
- Data is preprocessed before being passed to the class

Usage:

1. Import the class into your Python script or Jupyter Notebook

   ```
   from neural_network import ArtificialNeuralNetwork
   ```

2. Create an instance of the class

   ```
   ann = ArtificialNeuralNetwork()
   ```

3. Load the data from the CSV file and map the class labels to integers

   ```
   ann.loadFile(file_path, column_names)
   ```

4. Scale the data using the `scale_data` method

   ```
   ann.scale_data()
   ```

5. Split the data into training and validation sets using the `split` method

   ```
   train_data, val_data = ann.split(frac=0.8, random_state=1)
   ```

6. Train the neural network using the `train` method

   ```
   ann.train(train_data, num_epochs=1000, learning_rate=0.1, num_hidden=5)
   ```

7. Test the neural network on the validation set using the `test` method

   ```
   accuracy = ann.test(val_data)
   ```

8. Predict the class label for a new data point using the `predict` method

   ```
   predicted_class = ann.predict(new_data_point)
   ```

Note: This implementation assumes that the data is already preprocessed and cleaned before being passed to the class. It also assumes that the class labels are mutually exclusive and there are no missing values in the dataset. If your data does not meet these assumptions, you may need to modify the code accordingly.

Example:

```
from neural_network import ArtificialNeuralNetwork
import pandas as pd

# Load the data from a CSV file
data = pd.read_csv('iris.csv')

# Define the column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Create an instance of the neural network class
ann = ArtificialNeuralNetwork()

# Load the data and map the class labels to integers
ann.loadFile('iris.csv', column_names)

# Scale the data
ann.scale_data()

# Split the data into training and validation sets
train_data, val_data = ann.split(frac=0.8, random_state=1)

# Train the neural network
ann.train(train_data, num_epochs=1000, learning_rate=0.1, num_hidden=5)

# Test the neural network on the validation set
accuracy = ann.test(val_data)

# Predict the class label for a new data point
new_data_point = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]])
predicted_class = ann.predict(new_data_point)
```