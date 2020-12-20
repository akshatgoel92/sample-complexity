# Import packages
import os
import pickle
import helpers
import numpy as np 
import matplotlib.pyplot as plt

# Potential report content
# Talk about effect of dimensionality on overfitting
# Expand the Gaussian kernel into its feature map and speak about the role of c as a regularizer
# Try to answer the question: for what values of C does Gaussian kernel mimic a polynomial kernel? 
# Potentially make a plot for the above

# Questions: 
# Do we shuffle the data at each epoch?



def train_perceptron(X_train, Y_train, 
                     X_val, Y_val, epochs, 
                     kernel_type, d, n_classes, 
                     cf = False, fit_type = 'one_vs_all', 
                     convergence_epochs=10):
    '''
    --------------------------------------
    This is the main training loop for
    the kernel perceptron algorithm.
    --------------------------------------
    '''
    # Store a record of training and validation accuracies and other data from each epoch
    history = {
        "train_accuracies": [],
        "val_accuracies": [],
        "train_cf": [],
        "val_cf": []
    }

    # Store minimum loss for convergence check
    max_accuracy = 0
    convergence_counter = 0
    
    # Transform X according to the user specified kernel
    # Can be either polynomial kernel or Gaussian kernel
    # Do this for both training and validation set
    if kernel_type == 'polynomial':
        K_train = helpers.get_polynomial_kernel(X_train, X_train, d)
        K_val = helpers.get_polynomial_kernel(X_train, X_val, d)
    
    elif kernel_type == 'gaussian':
        K_train = helpers.get_gaussian_kernel(X_train, X_train, d)
        K_val = helpers.get_gaussian_kernel(X_train, X_val, d)

    # Store encoding
    # Can be encoding according to one vs all or all pairs
    if fit_type == 'one_vs_all':
        Y_encoding = helpers.get_one_vs_all_encoding(Y_train, n_classes)
    elif fit_type == 'all_pairs':
        Y_encoding = helpers.get_all_pairs_encoding(Y_train, n_classes)
    
    # Initialize alpha weights and store 
    # the number of samples
    alpha = np.zeros((n_classes, K_train.shape[0]))
    n_samples = np.max(Y_train.shape)

    # Run for a fixed user-specified number of epochs
    for epoch in range(epochs):

        if convergence_counter >= convergence_epochs:
            break
        
        # Print the epoch number to track progress
        print('This is epoch number: {}'.format(epoch))

        # Do this for each example in the dataset
        for i in range(n_samples):
            # Compute the prediction with the current weights:
            # dim(alpha) --> (10, 6199), 
            # dim(K_train[i, :]) ---> (6199, 1) 
            # ====> dim(y_hat) --> 10 X 1
            Y_hat, _ = helpers.get_predictions(alpha, K_train[i, :])
            
            # Perform update by calling the function above
            alpha = get_update(Y_train, Y_hat, alpha, n_classes, i, Y_encoding)
            
        # We finally compute predictions and accuracy at the end of each epoch
        # It is a mistake if the class with the highest predicted value does not equal the true label
        # mistakes += int((np.argmax(Y_hat) + 1) != int(Y_train[i]))
        Y_hat_train, preds_train = helpers.get_predictions(alpha, K_train)
        train_accuracy = helpers.get_accuracy(Y_train, preds_train)
            
        # Now we compute validation predictions
        Y_hat_val, preds_val = helpers.get_predictions(alpha, K_val)
        val_accuracy = helpers.get_accuracy(Y_val, preds_val)

        if helpers.has_improved(max_accuracy, val_accuracy):
            max_accuracy = val_accuracy
        else:
            convergence_counter +=1
        
        # We append to the history dictionary as a record
        history['train_accuracies'].append(train_accuracy)
        history['val_accuracies'].append(val_accuracy)

        if cf: 
        
            # At the end of each epoch we get confusion matrices
            train_cf = helpers.get_confusion_matrix(Y_train, preds_train)
            val_cf = helpers.get_confusion_matrix(Y_val, preds_val)
            history['train_cf'].append(train_cf)
            history['val_cf'].append(val_cf)

        
        # We print the accuracies at the end of each epoch
        msg = '{} accuracy on epoch {}: {}'
        print(msg.format('Train', epoch, train_accuracy))
        print(msg.format('Validation', epoch, val_accuracy))
    
    # Return statement
    return(history)


def get_update(Y_train, Y_hat, alpha, n_classes, i, Y):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.

    # Now first make a matrix Y with dim(Y) ---> (n_classes,) which is only filled with -1
    # Then get the label from the Y_train matrix
    # If this label is 6 then we want to change the 6th index to 1
    # Y = get_one_hot_encoding(n_classes, Y_train, i)
    # Compute sign of predictions and store indices to update        
    # Check if the prediction is correct against the labels
    # If it is correct we don't need to make any updates: we just move to the next iteration
    # If it is not correct then we update the weights and biases in the direction of the label
    --------------------------------------
    '''
    Y = Y[i]
    signs = helpers.get_signs(Y_hat, Y)
    wrong = (Y*Y_hat <= 0)
    alpha[wrong, i] -= (signs[wrong])
    
    return(alpha)



def run_test_case(epochs, kernel_type, d, n_classes):
    '''
    --------------------------------------
    Execute the training steps above and generate
    the results that have been specified in the report.
    --------------------------------------
    '''
    X_train, Y_train = helpers.load_data("data", "dtrain123.dat")
    X_val, Y_val = helpers.load_data("data", "dtest123.dat")

    Y_train = Y_train - 1
    Y_val = Y_val - 1

    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    history = train_perceptron(X_train, Y_train, X_val, Y_val, epochs, 
                               kernel_type, d, n_classes)

    return(history)



def run_multiple(params, data_args, kwargs, total_runs=5):
    '''
    --------------------------------------
    Run multiple runs of kernel 
    perceptron training with a given 
    set of parameters
    --------------------------------------
    '''
    results = []
    
    for param in params:

        histories = {
        
        'params': param, 
        'history': [],
        'best_epoch': [],
        'best_training_accuracy': [],
        'best_dev_accuracy': [],
        }
        
        for run in range(total_runs):
            # Prepare data for the perceptron
            # Shuffle the dataset before splitting it
            # Split the data into training and validation set 
            X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
            X, Y = helpers.shuffle_data(X, Y)
            
            X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, data_args['train_percent'])
            Y_train = Y_train.astype(int)
            Y_val = Y_val.astype(int)

            # Call the perceptron training with the given epochs
            # Return best epoch according to dev. accuracy and the associated accuracies on both datasets
            history = train_perceptron(X_train, Y_train, X_val, Y_val, **kwargs, d=param)
            best_epoch, best_training_accuracy, best_dev_accuracy = helpers.get_best_results(history)
            
            # Store results
            histories['best_training_accuracy'].append(best_training_accuracy)
            histories['best_dev_accuracy'].append(best_dev_accuracy)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(history)
        
        # Store results
        results.append(histories)
    
    helpers.get_experiment_results(results, '3_1')
    
    return(histories)



def run_k_fold_cross_val(epochs, data_path, name, kernel_type, 
                         d, n_classes, k):
    '''
    --------------------------------------
    Execute the training steps above and generate
    the results that have been specified in the report.
    --------------------------------------
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
        history = train_perceptron(X_train, Y_train, 
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



def run_grid_search(params, kwargs):
    '''
    --------------------------------------
    Check which kernel parameter results
    in highest validation accuracy
    --------------------------------------
    '''
    get_histories = {
        
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

    run_test = 1
    run_mul = 1
    run_cv = 0

    # How many runs to do for each hyper-parameter value?
    total_runs = 2

    # Store test arguments
    test_args = {

    'kernel_type': 'polynomial',
    'n_classes': 3,
    'epochs':20,
    'd':3,

    }

    # Store kernel parameter list to iterate over
    params = [1, 2]

    data_args = {

        'data_path': 'data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8

    }

    # Store arguments for this
    multiple_run_args = {
    
        'epochs': 5, 
        'kernel_type': 'polynomial', 
        'n_classes': 10,
        'convergence_epochs': 3   
    }


    # Search for the best parameters with the polynomial kernel
    grid_search_args = {
    
        'epochs': 4,
        'kernel_type': 'gaussian', 
        'n_classes': 10,
        'k': 5
    
    }


    if run_test == 1:
        # Call test function
        history = run_test_case(**test_args)

    if run_mul == 1:
        # Call training function multiple runs
        multiple_histories = run_multiple(params, data_args, multiple_run_args, total_runs)

    if run_cv == 1:
        # Call training function with k-fold cross validation
        cross_val_histories = run_grid_search(params, grid_search_args)