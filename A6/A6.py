from ArtificialNeuralNetwork import *


# Create an instance of the ArtificialNeuralNetwork class
myANN = ArtificialNeuralNetwork()

# Load the Iris data file and set the column names
myANN.loadFile('./ANN - Iris data.txt', col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

# Scale the data (out of place)
scaled_data = myANN.scale_data()


# Split the data into training and validation sets
train_data, val_data = myANN.split()


# Train the model using the training set
myANN.train(train_data)

# Test the model using the validation set and print the accuracy
accuracy = myANN.test(val_data)
print("Accuracy:", accuracy)

def prompt_query():
    """
    Input:

        None

    Output:

        A Pandas DataFrame with scaled input data from user input.

    Description:

        - This function prompts the user to enter the sepal length, sepal width, petal length, and petal width of an iris flower.
        - The user inputs are then used to create a Pandas DataFrame with the input data.
        - The input data is then scaled using a function from an object of the ArtificialNeuralNetwork class, named myANN.
        - The function returns the scaled input data as a Pandas DataFrame.
    """
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    # create input data as a DataFrame
    input_data = pd.DataFrame({
        "sepal_length": [sepal_length],
        "sepal_width": [sepal_width],
        "petal_length": [petal_length],
        "petal_width": [petal_width],
        "class": "tbd"
    })
    # scale input data
    scaled_input_data = myANN.scale_row(input_data)
    return scaled_input_data

# Create a dictionary to map the predicted class integer to the corresponding class name
int_to_class = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}


# Prompt the user to enter input and predict the class until the user chooses to exit
while True:
    flag = input("Do you want to query? (y/n)")
    if flag == "n":
        break
    elif flag != "y":
        print("only (y/n) are valid input")
        continue
    # Prompt the user for input and get the scaled input data
    input_data = prompt_query()
    # Predict the class and print the result
    print("The model predicts this flower is of type",int_to_class[myANN.predict(input_data)])
