import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

class Regressor():

    def __init__(self, x, nb_epoch = 1000, learning_rate = 0.01, hidden_size = 4):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.x_label_binarizer = LabelBinarizer()
        self.x_numerical_scaler = MinMaxScaler()
        self.y_numerical_scaler = MinMaxScaler()
        
        # Initialize model parameters
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.hidden_size = hidden_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate

        # Xavier initialization for weights between input and hidden layer
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1 / self.input_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))

        # Xavier initialization for weights between hidden and output layer
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        return


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        categorical_columns = x.select_dtypes(include=['object']).columns
        numerical_columns = x.select_dtypes(include=['float64', 'int64']).columns

        # Make a copy of x and y (if applicable) to avoid modifying the original data
        x_copy = x.copy()
        y_copy = None if (y is None) else y.copy()
        
        # Fill missing values in categorical columns with the mode
        for column in categorical_columns:
            x_copy[column].fillna(x_copy[column].mode()[0], inplace=True)

        # Fill missing values in numeric columns with the mean
        for column in numerical_columns:
            x_copy[column].fillna(x_copy[column].mean(), inplace=True)
    
        # Preprocess x based on training or testing
        if training:
            # Create a new one-hot encoder and scaler as this is a training dataset
            self.x_label_binarizer = LabelBinarizer()
            self.x_numerical_scaler = MinMaxScaler()

            # Normalise numerical values
            x_copy[numerical_columns] = self.x_numerical_scaler.fit_transform(x_copy[numerical_columns])

            # One-hot encode categorical values
            for column in categorical_columns:
                # Fit the label binarizer on the categorical column and create a new dataframe with the transformed columns
                transformed_categorical_cols = self.x_label_binarizer.fit_transform(x_copy[column])
                transformed_cols_df = pd.DataFrame(transformed_categorical_cols, columns=self.x_label_binarizer.classes_)

                # Concatenate the transformed columns to the original dataframe and remove the original categorical column
                x_copy = pd.concat([x_copy.reset_index(drop=True), transformed_cols_df.reset_index(drop=True)], axis=1)
                x_copy.drop(column, axis=1, inplace=True)

        else:
            # Normalise numerical values
            x_copy[numerical_columns] = self.x_numerical_scaler.transform(x_copy[numerical_columns])

            # One-hot encode categorical values
            for column in categorical_columns:
                # Transform the categorical column and create a new dataframe with the transformed columns
                transformed_categorical_cols = self.x_label_binarizer.transform(x_copy[column])
                transformed_cols_df = pd.DataFrame(transformed_categorical_cols, columns=self.x_label_binarizer.classes_)
                
                # Concatenate the transformed columns to the original dataframe and remove the original categorical column
                x_copy = pd.concat([x_copy.reset_index(drop=True), transformed_cols_df.reset_index(drop=True)], axis=1)
                x_copy.drop(column, axis=1, inplace=True)

        # Preprocess y if necessary
        if y_copy is not None:
            if training:
                # Create a new scaler as this is a training dataset
                self.y_numerical_scaler = MinMaxScaler()

                # Normalise numerical values
                y_copy = self.y_numerical_scaler.fit_transform(y_copy if isinstance(y_copy, pd.DataFrame) else y_copy.values.reshape(-1, 1))
            else:
                # Normalise numerical values
                y_copy = self.y_numerical_scaler.transform(y_copy if isinstance(y_copy, pd.DataFrame) else y_copy.values.reshape(-1, 1))

        # Convert x_copy to torch tensor
        x_copy = torch.tensor(x_copy.values)

        return x_copy, y_copy


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y, batch_size=32, shuffle=True):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).
            - batch_size {int} -- Size of the batch for training.
            - shuffle {bool} -- Whether to shuffle the data before training.

        Returns:
            self {Regressor} -- Trained model.
        """

        X, Y = self._preprocessor(x, y=y, training=True)  # Preprocess the data

        # Get the number of data points
        data_size = X.shape[0]
        
        for _ in range(self.nb_epoch):
            if shuffle:
                # Shuffle the data
                indices = np.random.permutation(data_size)
                X = X[indices]
                Y = Y[indices]

            # Iterate over batches
            for i in range(0, data_size, batch_size):
                # Get the current batch
                x_batch = X[i:i + batch_size]
                y_batch = Y[i:i + batch_size]

                forward_pass_output = self.forward_pass(x_batch)
                predictions = forward_pass_output['predictions']
                hidden_layer_output = forward_pass_output['hidden_layer_output']
    
                loss = self.calculate_loss(predictions, y_batch)
                gradients = self.backward_pass(x_batch, hidden_layer_output, predictions, y_batch)
                self.update_parameters(gradients)

        return self


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def calculate_loss(self, predictions, y):
        # Calculate Mean Squared Error loss
        loss = np.mean((predictions - y) ** 2)
        return loss

    def forward_pass(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_input_hidden
        hidden_layer_output = self._sigmoid(hidden_layer_input)

        # The output layer input is the prediction as there is no activation function on the output layer
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        
        return {'predictions': output_layer_input,
                'hidden_layer_output': hidden_layer_output}

    def backward_pass(self, x, hidden_layer_output, predictions, y):
        # Calculate the error
        output_error = 2 * (predictions - y) / len(y)

        # Backpropagate the error
        d_output = output_error
        d_hidden_layer = np.dot(d_output, self.weights_hidden_output.T) * self._sigmoid_derivative(hidden_layer_output)

        # Calculate the gradients
        gradients_hidden_output = np.dot(hidden_layer_output.T, d_output)
        gradients_input_hidden = np.dot(x.T, d_hidden_layer)

        return {'weights_input_hidden': gradients_input_hidden,
                'weights_hidden_output': gradients_hidden_output,
                'bias_input_hidden': np.sum(d_hidden_layer, axis=0, keepdims=True),
                'bias_hidden_output': np.sum(d_output, axis=0, keepdims=True)}

    def update_parameters(self, gradients):
        self.weights_input_hidden -= self.learning_rate * gradients['weights_input_hidden']
        self.weights_hidden_output -= self.learning_rate * gradients['weights_hidden_output']
        self.bias_input_hidden -= self.learning_rate * gradients['bias_input_hidden']
        self.bias_hidden_output -= self.learning_rate * gradients['bias_hidden_output']

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        # Forward pass to get the scaled predictions
        raw_predictions = self.forward_pass(X)['predictions']

        # Inverse transform the predictions to get the original values
        predictions = self.y_numerical_scaler.inverse_transform(raw_predictions)
    
        return predictions


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_output = self.predict(x)
        return r2_score(y,y_output)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
