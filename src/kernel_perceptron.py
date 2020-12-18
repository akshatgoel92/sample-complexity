#!/usr/bin/env python
# coding: utf-8

# Import packages
import os
import numpy as np 
import matplotlib.pyplot as plt
import helpers





def get_predictions(alpha, K_examples):
    '''
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    '''
    # Take the maximum argument in each column
    Y_hat = alpha @ K_examples
    preds = np.argmax(Y_hat, axis = 0)
    
    # Return statement
    return(Y_hat, preds)


def get_update(Y_train, Y_hat, alpha, n_classes, i):
    '''
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    '''
    # Now first make a matrix Y with dim(Y) ---> (n_classes,) which is only filled with -1
    # Then get the label from the Y_train matrix
    # If this label is 6 then we want to change the 6th index to 1
    Y = np.full(n_classes, -1)
    Y[int(Y_train[i])] = 1
            
    # Compute sign of predictions
    # This is used in the update
    signs = np.ones(Y_hat.shape)
    signs[Y_hat <= 0] = -1
            
    # Check if the prediction is correct against the labels
    # If it is correct we don't need to make any updates: we just move to the next iteration
    # If it is not correct then we update the weights and biases in the direction of the label
    alpha[Y*Y_hat <= 0, i] -= (signs[Y*Y_hat <= 0])
    
    return(alpha)


def train_kernel_perceptron(X_train, Y_train, X_val, Y_val, epochs, kernel_type, d, n_classes):
    '''
    This is the main training loop for
    the kernel perceptron algorithm.
    '''
    # Store a record of training and validation accuracies and other data from each epoch
    history = {
        "train_accuracies": [],
        "val_accuracies": [],
        "train_cf": [],
        "val_cf": []
    }
    
    # Transform X according to the user specified kernel
    # Can be either polynomial kernel or Gaussian kernel
    # Do this for both training and validation set
    if kernel_type == 'polynomial':
        K_train = helpers.get_polynomial_kernel(X_train, X_train, d)
        K_val = helpers.get_polynomial_kernel(X_train, X_val, d)
    
    elif kernel_type == 'gaussian':
        K_train = helpers.get_gaussian_kernel(X_train, X_train, d)
        K_val = helpers.get_gaussian_kernel(X_train, X_val, d)
    
    # Initialize alpha weights and store 
    # the number of samples
    alpha = np.zeros((n_classes, K_train.shape[0]))
    n_samples = max(Y_train.shape)
    
    # Run for a fixed user-specified number of epochs
    for epoch in range(epochs):
        
        # Print the epoch number to track progress
        print('This is epoch number: {}'.format(epoch))

        # Do this for each example in the dataset
        for i in range(n_samples):
            # Compute the prediction with the current weights:
            # dim(A) --> (10, 6199), dim(X_train[i, :]) ---> (6199, 1) ====> dim(y_hat) --> 10 X 1
            Y_hat, _ = get_predictions(alpha, K_train[i, :])
            
            # Perform update by calling the function above
            alpha = get_update(Y_train, Y_hat, alpha, n_classes, i)
            
        # We finally compute predictions and accuracy at the end of each epoch
        # It is a mistake if the class with the highest predicted value does not equal the true label
        # mistakes += int((np.argmax(Y_hat) + 1) != int(Y_train[i]))
        Y_hat_train, preds_train = get_predictions(alpha, K_train)
        train_accuracy = helpers.get_accuracy(Y_train, preds_train)
            
        # Now we compute validation predictions
        Y_hat_val, preds_val = get_predictions(alpha, K_val)
        val_accuracy = helpers.get_accuracy(Y_val, preds_val)
        
        # At the end of each epoch we get confusion matrices
        train_cf = helpers.get_confusion_matrix(Y_train, preds_train)
        val_cf = helpers.get_confusion_matrix(Y_val, preds_val)
        
        # We append to the history dictionary as a record
        history['train_accuracies'].append(train_accuracy)
        history['val_accuracies'].append(val_accuracy)
        history['train_cf'].append(train_cf)
        history['val_cf'].append(val_cf)
        
        # We print the accuracies at the end of each epoch
        msg = '{} accuracy on epoch {}: {}'
        print(msg.format('train', epoch, train_accuracy))
        print(msg.format('validation', epoch, val_accuracy))
    
    # Return statement
    return(history)


def run_perceptron_training(epochs, data_path = os.path.join('..', 'data'), name = 'zipcombo.dat', 
                            kernel_type = 'polynomial', d = 5, n_classes=10, train_percent=0.8):
    '''
    Execute the training steps above and generate
    the results that have been specified in the report.
    '''
    # Prepare data for the perceptron
    X, Y = helpers.load_data(data_path, name)
    
    # Shuffle the dataset before splitting it
    X, Y = helpers.shuffle_data(X, Y)
    
    # Split the data into training and validation set 
    X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, train_percent)

    # Call the perceptron training with the given epochs
    history = train_kernel_perceptron(X_train, Y_train, 
                                      X_val, Y_val, epochs,
                                      kernel_type, d, n_classes)
    
    # Return best epoch according to dev. accuracy and the associated accuracies on both datasets
    best_epoch, best_training_accuracy, best_dev_accuracy = helpers.get_best_results(history)
    
    # Return statement
    return(history, best_epoch, best_training_accuracy, best_dev_accuracy)

def run_k_fold_cross_val(epochs, data_path = os.path.join('..', 'data'), 
                         name = 'zipcombo.dat', kernel_type = 'polynomial', 
                         d = 5, n_classes=10, k = 5):
    '''
    Execute the training steps above and generate
    the results that have been specified in the report.
    '''
    # Prepare data for the perceptron
    X, Y = helpers.load_data(data_path, name)
    
    # Shuffle the dataset before splitting it
    X, Y = helpers.shuffle_data(X, Y)
    
    # Split the data into training and validation set 
    X_folds, Y_folds = helpers.get_k_folds(X, Y, k)
    
    # Initiate histories object
    histories = []
    
    # Now go through each fold : every fold becomes the hold-out set at least once
    for fold_no in range(k):
        
        # Put in the x-values
        X_train = np.concatenate(X_folds[:fold_no] + X_folds[fold_no+1:])
        X_val = X_folds[fold_no]
        
        # Put in the Y values
        Y_train = np.concatenate(Y_folds[:fold_no] + Y_folds[fold_no+1:])
        Y_val =  Y_folds[fold_no]
        
        # Call the perceptron training with the given epochs
        history = train_kernel_perceptron(X_train, Y_train, 
                                          X_val, Y_val, epochs,
                                          kernel_type, d, n_classes)
        
        # Append to the histories file the epoch by epoch record of each fold
        histories.append(history)
    
    # Get avg. accuracies by epoch across folds
    avg_history = helpers.get_avg_results(histories)
    
    # Return best epoch according to dev. accuracy and the associated accuracies on both datasets
    best_epoch, best_training_accuracy, best_dev_accuracy = helpers.get_best_results(avg_history)
        
    # Return statement
    return(avg_history, best_epoch, best_training_accuracy, best_dev_accuracy)




def run_multiple(params, kwargs):
    
    histories = {
        
        'params': params, 
        'history': [],
        'best_epoch': [],
        'best_training_accuracy': [],
        'best_dev_accuracy': [],
    }
    
    for param in params:
        history, best_epoch, best_training_accuracy, best_dev_accuracy = run_perceptron_training(**kwargs, d=param)
        histories['history'].append(history)
        histories['best_epoch'].append(best_epoch)
        histories['best_training_accuracy'].append(best_training_accuracy)
        histories['best_dev_accuracy'].append(best_dev_accuracy)
    
    return(histories)



def run_grid_search(params, kwargs):
    
    
    histories = {
        
            'params': params, 
            'best_epoch': [],
            'best_training_accuracy': [],
            'best_dev_accuracy': [],
    }
    
    for param in params: 
        _, best_epoch, best_training_accuracy, best_dev_accuracy = run_k_fold_cross_val(d=param, **kwargs)
        histories['best_epoch'].append(best_epoch)
        histories['best_training_accuracy'].append(best_training_accuracy)
        histories['best_dev_accuracy'].append(best_dev_accuracy)
    
    
    return(histories)



if __name__ == '__main__':

    # Set random seed
    np.random.seed(13290138)
    
    # Load data
    X_train, Y_train = helpers.load_data(os.path.join("..", "data"), "dtrain123.dat")
    X_val, Y_val = helpers.load_data(os.path.join("..", "data"), "dtest123.dat")

    Y_train = Y_train - 1
    Y_val = Y_val - 1

    epochs=100
    n_classes=3
    kernel_type = 'polynomial'
    d=3

    history = train_kernel_perceptron(X_train, Y_train, 
                                  X_val, Y_val, epochs,
                                  kernel_type, d, n_classes)





    # Store parameter list
    params = [1, 2, 3, 4, 5, 6, 7]


    # Store arguments for this
    multiple_run_args = {
    
        'epochs': 100, 
        'data_path': os.path.join('..', 'data'), 
        'name': 'zipcombo.dat', 
        'kernel_type': 'polynomial', 
        'n_classes': 10,
        'train_percent': 0.8   
    }

    # Call training function multiple runs
    multiple_histories = run_multiple(params, multiple_run_args)


    # Search for the best parameters with the polynomial kernel
    grid_search_args = {
    
        'epochs': 100, 
        'data_path': os.path.join('..', 'data'), 
        'name': 'zipcombo.dat', 
        'kernel_type': 'polynomial', 
        'n_classes': 10,
        'k': 5
    
    }

    # Call training function with k-fold cross validation
    cross_val_histories = run_grid_search(params, grid_search_args)


    # To do for this script:
    # Make plots of losses to see whether it is converging
    # Check shapes of all matrices
    # Take a look at the actual data in each matrix
    # Take a look at each weight
    # Take a look at whether kernel arguments are being sent in correctly
    # Test Gaussian kernel

    # Potential report content
    # Talk about effect of dimensionality on overfitting
    # Expand the Gaussian kernel into its feature map and speak about the role of c as a regularizer
    # Try to answer the question: for what values of C does Gaussian kernel mimic a polynomial kernel? 
    # Potentially make a plot for the above

    # Questions: 
    # Do we shuffle the data at each epoch?