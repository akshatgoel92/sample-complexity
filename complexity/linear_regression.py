# Import packages
import os
import numpy as np
import helpers


def train_regression(X_train, Y_train, X_val, Y_val):
    '''
    ------------------------
    Input: k: Dimension of basis
           x: Feature values
           y: Labels
    Output: Results from running
    polynomial basis regression of
    dimension k on x and y
    ------------------------
    ''' 
    beta_hat = np.linalg.pinv(X_train) @ Y_train
    
    y_hat_train = np.sign(X_train @ beta_hat)
    y_hat_train[y_hat_train == 0] = -1

    y_hat_val = np.sign(X_val @ beta_hat)
    y_hat_val[y_hat_val == 0] = -1
    
    train_loss = helpers.get_loss(Y_train, y_hat_train)
    val_loss = helpers.get_loss(Y_val, y_hat_val)
    
    return(train_loss, val_loss)



def get_regression(m, n):
    '''
    --------------------
    Run linear regression algorithm to get a base-line
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
    X, Y = helpers.get_binary_data(m, n)

    # 
    X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, 0.8)
    
    # Call the perceptron training with the given epochs
    history = train_regression(X_train, Y_train, X_val, Y_val)
    
    # Return statement
    return(history)


if __name__ == '__main__':
    
    np.random.seed(102938120)

    # Set parameters
    m = 10
    n = 1000
    
    # Call training function
    history = get_regression(m, n)
    print(history)