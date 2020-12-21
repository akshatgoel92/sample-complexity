# Import packages
import os
import time
import pickle
import helpers
import numpy as np
import scipy.sparse as sparse 
import matplotlib.pyplot as plt

# Checks to do:
# 1) Check CV method of averaging
# 2) Change from accuracy to error everywhere
# 3) Check how best to represent sum 
# 4) Check how best to represent 

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
                     tolerance=0.00001, convergence_epochs=5):
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
        "preds_train": [],
        "preds_val": []
    }

    # Store minimum loss for convergence check
    prev_accuracy = 0
    convergence_counter = 0

    # Initialize alpha weights and store 
    # the number of samples


    # Store encoding
    Y_encoding = helpers.get_one_vs_all_encoding(Y_train, n_classes)
    
    # Transform X according to the user specified kernel
    # Can be either polynomial kernel or Gaussian kernel
    # Do this for both training and validation set
    if kernel_type == 'polynomial':
        K_train = helpers.get_polynomial_kernel(X_train, X_train, d)
        K_val = helpers.get_polynomial_kernel(X_train, X_val, d)
    
    elif kernel_type == 'gaussian':
        K_train = helpers.get_gaussian_kernel(X_train, X_train, d)
        K_val = helpers.get_gaussian_kernel(X_train, X_val, d)

    alpha = np.zeros((n_classes, K_train.shape[0]))
    active = [np.array([], dtype=int) for row in range(alpha.shape[0])]
    n_samples = np.max(Y_train.shape)

    # Run for a fixed user-specified number of epochs
    for epoch in range(epochs):

        if convergence_counter >=convergence_epochs:
            break
        
        # Print the epoch number to track progress
        print('This is epoch number: {}'.format(epoch))
        
        # Do this for each example in the dataset
        for i in range(n_samples):
            # Compute the prediction with the current weights:
            # dim(alpha) --> (10, 6199), 
            # dim(K_train[i, :]) ---> (6199, 1) 
            # ====> dim(y_hat) --> 10 X 1
            # Perform update
            Y_hat, _ = get_predictions(alpha, K_train[i, :])
            # Y_hat, _ = get_sparse_predictions(alpha, K_train[i, :], active)
            alpha = get_update(Y_train, Y_hat, alpha, n_classes, i, Y_encoding, active)

        # We finally compute predictions and accuracy at the end of each epoch
        # It is a mistake if the class with the highest predicted value does not equal the true label
        # mistakes += int((np.argmax(Y_hat) + 1) != int(Y_train[i]))
        Y_hat_train, preds_train = get_predictions(alpha, K_train)
        train_accuracy = helpers.get_accuracy(Y_train, preds_train)
            
        # Now we compute validation predictions
        Y_hat_val, preds_val = get_predictions(alpha, K_val)
        val_accuracy = helpers.get_accuracy(Y_val, preds_val)


        # Convergence check
        if train_accuracy - prev_accuracy < tolerance:
            convergence_counter += 1

        # Update the previous accuracy after checking convergence
        prev_accuracy = train_accuracy
            
        
        # We append to the history dictionary as a record
        history['train_accuracies'].append(train_accuracy)
        history['val_accuracies'].append(val_accuracy)
        history['preds_train'].append(preds_train)
        history['preds_val'].append(preds_val)

        
        # We print the accuracies at the end of each epoch
        msg = 'Train accuracy: {}, Validation accuracy: {}, Epoch: {}'
        print(msg.format(train_accuracy, val_accuracy, epoch))
    
    # Return statement
    return(history)


def get_update(Y_train, Y_hat, alpha, n_classes, i, Y, active):
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
    signs = np.ones(Y_hat.shape)
    signs[Y_hat <= 0] = -1
    signs[Y == 0] = 0
    wrong = (Y*Y_hat <= 0)
    # wrong_indices = [i for i, result in enumerate(wrong) if result == True]

    if np.sum(wrong) > 0:
        alpha[wrong, i] -= signs[wrong]
        # active = [np.append(active[wrong_index], i) for wrong_index in wrong_indices]
    
    return(alpha)


def get_sparse_predictions(alpha, K_examples, active):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    --------------------------------------
    '''
    Y_hat = [alpha[class_no][is_active] @ K_examples[is_active] for class_no, is_active in enumerate(active)]
    Y_hat = np.array(Y_hat).reshape(alpha.shape[0], )
    preds = np.argmax(Y_hat, axis = 0)

    return(Y_hat, preds)
    
def get_predictions(alpha, K_examples):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    --------------------------------------
    '''
    # Take the maximum argument in each column
    Y_hat = alpha @ K_examples
    preds = np.argmax(Y_hat, axis = 0)
    
    # Return statement
    return(Y_hat, preds)


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
    
    helpers.save_experiment_results(results, '3_1')
    
    return(histories)



def run_multiple_cv(params, data_args, kwargs, total_runs=2):
    '''
    --------------------------------------
    Check which kernel parameter results
    in highest validation accuracy
    --------------------------------------
    '''
    results = []
    
    for run in range(total_runs):

        histories = {
        
            'params': params, 
            'history': [],
            'best_epoch': [],
            'best_training_accuracy': [],
            'best_dev_accuracy': [],
            }

        # Prepare data for the perceptron
        X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
        X, Y = helpers.shuffle_data(X, Y)

        X_train, X_test, Y_train, Y_test = helpers.split_data(X, Y, data_args['train_percent'])
        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)
            
        X_folds, Y_folds = helpers.get_k_folds(X_train, Y_train, data_args['k'])

        for param in params: 

            fold_histories = []
    
            # Now go through each fold : every fold becomes the hold-out set at least once
            for fold_no in range(data_args['k']):
        
                # Put in the x-values
                X_train_fold = np.concatenate(X_folds[:fold_no] + X_folds[fold_no+1:])
                X_val_fold = X_folds[fold_no]
        
                # Put in the Y values
                Y_train_fold = np.concatenate(Y_folds[:fold_no] + Y_folds[fold_no+1:])
                Y_val_fold =  Y_folds[fold_no]
        
                # Call the perceptron training with the given epochs
                history = train_perceptron(X_train_fold, Y_train_fold, 
                                           X_val_fold, Y_val_fold, **cv_args, d=param)
        
                # Append to the histories file the epoch by epoch record of each fold
                fold_histories.append(history)
            
            # Get avg. accuracies by epoch across folds
            avg_history = helpers.get_cv_results(fold_histories)
            best_epoch, best_training_accuracy, best_dev_accuracy = helpers.get_best_results(avg_history)
            histories['best_training_accuracy'].append(best_training_accuracy)
            histories['best_dev_accuracy'].append(best_dev_accuracy)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(avg_history)

        # Get best parameter value
        best_dev_config = np.argmin(np.array(histories['best_dev_accuracy']))
        best_param = histories['params'][best_dev_config]

        # Retrain
        print("Retraining now...")
        print("The best parameter is {}....".format(best_param))

        # We are ready to retrain
        history = train_perceptron(X_train, Y_train, 
                                   X_test, Y_test, 
                                   **cv_args, d=best_param)
        
        # Get retraining results
        best_epoch, best_training_accuracy, best_dev_accuracy = helpers.get_best_results(history)
        preds_train = history['preds_train'][best_epoch]
        preds_val = history['preds_val'][best_epoch]
        
        # Update the results
        histories['best_training_accuracy'] = [best_training_accuracy]
        histories['best_dev_accuracy'] = [best_dev_accuracy]
        histories['best_epoch'] = [best_epoch]
        histories['params'] = best_param
        histories['history'] = [history]
        histories['train_cf'] = helpers.get_confusion_matrix(Y_train, preds_train)
        histories['val_cf'] = helpers.get_confusion_matrix(Y_test, preds_val)

        # Append the results
        results.append(histories)

    # Save the results
    helpers.save_results(results, '3_2')
    helpers.save_experiment_results(results, '3_2')

    return(results)



if __name__ == '__main__':

    # Set random seed
    np.random.seed(13290138)

    # Generic message for elapsed time used later
    time_msg = "Elapsed time is....{} minutes"
    run_test = 0
    run_mul = 1
    run_cv = 0

    # How many runs to do for each hyper-parameter value?
    total_runs = 20

    # Store test arguments
    test_args = {

    'kernel_type': 'polynomial',
    'n_classes': 3,
    'epochs':20,
    'd':3,
    'tolerance': 0.0001

    }

    # Store kernel parameter list to iterate over
    params = [1, 2, 3, 4, 5, 6, 7]

    data_args = {

        'data_path': 'data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5

    }

    # Store arguments for this
    multiple_run_args = {
    
        'epochs': 20, 
        'kernel_type': 'polynomial', 
        'n_classes': 10,
        'tolerance': 0.000001,
        'convergence_epochs': 5   
    }


    # Search for the best parameters with the polynomial kernel
    cv_args = {
    
        'epochs': 4,
        'kernel_type': 'polynomial', 
        'n_classes': 10, 
        'tolerance':0.000001,
        'convergence_epochs': 5
    
    }


    if run_test == 1:
        history = run_test_case(**test_args)

    if run_mul == 1:
        start = time.time()
        run_multiple(params, data_args, multiple_run_args, total_runs)
        elapsed = (time.time() - start)/60
        print(time_msg.format(elapsed))

    if run_cv == 1:
        run_multiple_cv(params, data_args, cv_args)