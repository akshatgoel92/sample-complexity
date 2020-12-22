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
# 2) Change from loss to error everywhere
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
                     kernel_type, d, n_classifiers, 
                     tolerance=0.000001, convergence_epochs=5, 
                     sparse_setting=0, max_epochs = 20, fit_type='one_vs_all'):
    '''
    --------------------------------------
    This is the main training loop for
    the kernel perceptron algorithm.
    --------------------------------------
    '''
    # Store a record of training and validation accuracies and other data from each epoch
    history = {
        "train_loss": [],
        "val_loss": [],
        "preds_train": [],
        "preds_val": []
    }

    # Store minimum loss for convergence check
    prev_loss = np.inf
    convergence_counter = 0

    # Store encoding
    if fit_type == 'one_vs_all':
        Y_encoding = helpers.get_one_vs_all_encoding(Y_train, n_classifiers)
    elif fit_type == 'one_vs_one': 
        Y_encoding = np.ones(len(Y_train), np.int32)
        Y_encoding[Y_train == j] = -1
    
    # Get kernel
    if kernel_type == 'polynomial':
        K_train = helpers.get_polynomial_kernel(X_train, X_train, d)
        K_val = helpers.get_polynomial_kernel(X_train, X_val, d)
    
    elif kernel_type == 'gaussian':
        K_train = helpers.get_gaussian_kernel(X_train, X_train, d)
        K_val = helpers.get_gaussian_kernel(X_train, X_val, d)

    # Initialize
    alpha = np.zeros((n_classifiers, K_train.shape[0]))
    n_samples = np.max(Y_train.shape)

    mistake_tracker = []
    
    
    # Run for a fixed user-specified number of epochs
    for epoch in range(epochs):

        if convergence_counter >= convergence_epochs or np.allclose(prev_loss, 0.0) or epoch > max_epochs:
            break

        # Count mistakes
        mistakes = 0
        update_counter = 0
        
        # Do this for each example in the dataset
        for i in range(n_samples):

            Y_hat, y_pred, signs, wrong = get_train_predictions(alpha, K_train[i, :], Y_encoding[i])
            
            if np.sum(wrong) > 0:
                mistake_tracker.append(i)
                alpha[wrong, i] -= signs[wrong]

            # Store mistakes
            mistakes += (y_pred != Y_train[i]).astype(int)

        mistake_tracker = list(set(mistake_tracker))
        print(len(mistake_tracker))

        # We finally compute predictions and loss at the end of each epoch
        # It is a mistake if the class with the highest predicted value does not equal the true label
        # mistakes += int((np.argmax(Y_hat) + 1) != int(Y_train[i]))
        # Now we compute validation predictions
        train_loss = mistakes/n_samples

        # Convergence check
        if np.abs(train_loss - prev_loss) < tolerance:
            convergence_counter += 1

        # Update the previous loss after checking convergence
        prev_loss = train_loss
            
        # We append to the history dictionary as a record
        history['train_loss'].append(train_loss)
        
        # We print the accuracies at the end of each epoch
        msg = 'Train loss: {}, Epoch: {}'
        print(msg.format(train_loss, epoch))

    # Test the classifier
    Y_hat_val, preds_val = get_val_predictions(alpha, K_val)
    val_loss = helpers.get_loss(Y_val, preds_val)
    history['val_loss'].append(val_loss)
    history['preds_val'].append(preds_val)
    
    # Return statement
    return(history)


def get_train_predictions(alpha, K_examples, Y_encoding, train=False):
    '''
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    '''
    # Take the maximum argument in each column
    Y_hat = alpha @ K_examples
    preds = np.argmax(Y_hat, axis = 0)
    
    # Store the sign of the predictions
    # Then figure out which predictions are wrong?
    signs = np.sign(Y_hat)
    signs[Y_hat == 0] = -1
    wrong = (Y_encoding*Y_hat <= 0)

    return(Y_hat, preds, signs, wrong)


def get_val_predictions(alpha, K_examples):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    --------------------------------------
    '''
    # Take the maximum argument in each column
    Y_hat = alpha @ K_examples
    preds = np.argmax(Y_hat, axis = 0)
    
    return(Y_hat, preds)


def run_test_case(epochs, kernel_type, d, n_classifiers, tolerance, sparse_setting):
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
                               kernel_type, d, n_classifiers, tolerance)

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
    overall_run_no = 0

    time_msg = "Elapsed time is....{} minutes"
    start = time.time()

    
    for param in params:

        histories = {
        
        'params': param, 
        'history': [],
        'best_epoch': [],
        'best_training_loss': [],
        'best_dev_loss': [],
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
            # Return best epoch according to dev. loss and the associated accuracies on both datasets
            history = train_perceptron(X_train, Y_train, X_val, Y_val, **kwargs, d=param)
            best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(history)
            
            # Store results
            histories['best_training_loss'].append(best_training_loss)
            histories['best_dev_loss'].append(best_dev_loss)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(history)

            overall_run_no += 1
            print("This is overall run no {}".format(overall_run_no))
            elapsed = (time.time() - start)/60
            print(time_msg.format(elapsed))
        
        # Store results
        results.append(histories)
    
    helpers.save_experiment_results(results, '3_1')
    
    return(histories)



def run_multiple_cv(params, data_args, kwargs, total_runs=2):
    '''
    --------------------------------------
    Check which kernel parameter results
    in highest validation loss
    --------------------------------------
    '''
    results = []

    
    for run in range(total_runs):


        histories = {
        
            'params': params, 
            'history': [],
            'best_epoch': [],
            'best_training_loss': [],
            'best_dev_loss': [],
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
            best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(avg_history)
            histories['best_training_loss'].append(best_training_loss)
            histories['best_dev_loss'].append(best_dev_loss)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(avg_history)

        # Get best parameter value
        best_dev_config = np.argmin(np.array(histories['best_dev_loss']))
        best_param = histories['params'][best_dev_config]

        # Retrain
        print("Retraining now...")
        print("The best parameter is {}....".format(best_param))

        # We are ready to retrain
        history = train_perceptron(X_train, Y_train, 
                                   X_test, Y_test, 
                                   **cv_args, d=best_param)
        
        # Get retraining results
        best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(history)
        preds_train = history['preds_train'][best_epoch]
        preds_val = history['preds_val'][best_epoch]
        
        # Update the results
        histories['best_training_loss'] = [best_training_loss]
        histories['best_dev_loss'] = [best_dev_loss]
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
    run_test = 0
    run_mul = 1
    run_cv = 0

    # How many runs to do for each hyper-parameter value?
    total_runs = 20

    # Store test arguments
    test_args = {

    'kernel_type': 'polynomial',
    'n_classifiers': 3,
    'epochs':20,
    'd':3,
    'tolerance': 0.0001,
    'sparse_setting': 0

    }

    # Store kernel parameter list to iterate over
    params = [1, 2, 3, 4, 5, 6, 7]

    data_args = {

        'data_path': 'data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5,

    }

    # Store arguments for this
    multiple_run_args = {
    
        'epochs': 20, 
        'kernel_type': 'polynomial', 
        'n_classifers': 10,
        'tolerance': 0.000001,
        'convergence_epochs': 5,
        'sparse_setting': 0,
        'tolerance': 0.0000001
    }


    # Search for the best parameters with the polynomial kernel
    cv_args = {
    
        'epochs': 4,
        'kernel_type': 'polynomial', 
        'n_classifiers': 10, 
        'tolerance':0.000001,
        'convergence_epochs': 5, 
        'sparse_setting': 0
    
    }

    if run_test == 0:
        history = run_test_case(**test_args)

    if run_mul == 1:
        run_multiple(params, data_args, multiple_run_args, total_runs)

    if run_cv == 1:
        run_multiple_cv(params, data_args, cv_args)