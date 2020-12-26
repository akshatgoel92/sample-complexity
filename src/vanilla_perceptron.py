# Import packages
from helpers import load_all_data, vectorized_flatten, sigmoid, get_log_loss, get_accuracy , shuffle_data
from helpers import sigmoid_derivative, gradient_update, plot_loss, prep_data
import numpy as np



def train_perceptron(X, Y, X_dev, y_dev, epochs, lr):
    '''
    --------------------
    Perceptron algorithm
    --------------------
    Parameters: 
    X: Numpy array of training features (shape = 784 X n)
    y: Binary (1/-1) training label (shape = n X 1)
    --------------------
    Output: 
    w: trained weights
    b: trained biases
    y_preds: predictions 
    --------------------
    '''
    # Initialize weights and biases
    w = np.zeros(X.shape[0])
    b = 0
    
    # History goes here
    history = {
        "weights": [w],
        "losses": [], 
        "biases": [b],
        "accuracies": [],
        "dev_accuracies": []
    }
    
    # Best accuracy
    best_accuracy = 0
    
    # Run for a fixed number of epochs
    for epoch in range(epochs):
        
        X, Y = shuffle_data(X, Y)
        
        # Do this for each example in the dataset
        for i in range(X.shape[1]):

            # Store the sample data
            x_i = X[:, i]
            y_i = Y[0][i]
            
            # Compute the prediction with the current weights
            if (np.dot(w, x_i) + b > 0): y_hat = 1
            else: y_hat = -1
            
            # Check if the prediction is correct against the labels
            # If it is correct we don't need to make any updates: we just move to the next iteration
            # If it is not correct then we do the following: 
            # 1) Update the weights and biases in the direction of the label
            if y_hat != y_i:
                w += lr*(y_i - y_hat)*x_i
                b += lr*(y_i - y_hat)
            
            
        # Get predictions on train and test
        y_train_preds = np.array([int(np.dot(w, X[:, i]) + b  > 0) for i in range(X.shape[1])])
        y_dev_preds = np.array([int(np.dot(w, X_dev[:, i]) + b  > 0) for i in range(X_dev.shape[1])])
        
        # Training accuracy                       
        accuracy = get_accuracy(Y, y_train_preds)
        dev_accuracy = get_accuracy(y_dev, y_dev_preds)
        print("Epoch {}/{}: Training_accuracy = {}, Dev. Accuracy = {}".format(epoch, epochs, accuracy, dev_accuracy))
         
        # Append results to history
        history["biases"].append(b)
        history["weights"].append(w)
        history["accuracies"].append(accuracy)
        history["dev_accuracies"].append(dev_accuracy)
        
        # Get training accuracy
        print("Epoch {}/{}: Training_accuracy = {}".format(epoch, epochs, accuracy))
    
    # Return statement
    return(history)


def get_perceptron_baseline(data_path, epochs, lr):
    '''
    --------------------
    Run perceptron algorithm to get a base-line
    --------------------
    Parameters: 
    X: Numpy array of training features (shape = 784 X n)
    y: Binary (1/0) training label (shape = n X 1)
    --------------------
    Output: 
    w: trained weights
    y_preds: predictions
    --------------------
    '''
    # Set the random seed for np.random number generator
    # This will make sure results are reproducible
    
    
    # Prepare data for the perceptron
    X_train_flattened, X_dev_flattened, X_test_flattened, y_train, y_dev, y_test = prep_data(data_path)
    
    # Call the perceptron training with the given epochs
    history = train_perceptron(X_train_flattened, y_train, X_dev_flattened, y_dev, epochs, lr)
    
    # Get results from history
    best_epoch, best_training_accuracy, best_dev_accuracy = get_results(history)
    
    # Return statement
    return(best_epoch, best_training_accuracy, best_dev_accuracy, history)


if __name__ == '__main__':

    np.random.seed(132089)
    
    # Set parameters
    data_path = '../setup/data'
    epochs = 1000
    lr = 0.6
    
    # Call training function
    best_epoch, best_accuracy, best_loss, history = get_perceptron_baseline(data_path, epochs, lr)

