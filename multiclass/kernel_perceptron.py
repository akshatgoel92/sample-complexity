# Import packages
import os
import time
import pickle
import helpers
import argparse
import numpy as np
import scipy.sparse as sparse 
import matplotlib.pyplot as plt

# Checks to do:
# 1) Check CV method of averaging

# Potential report content
# Talk about effect of dimensionality on overfitting
# Expand the Gaussian kernel into its feature map and speak about the role of c as a regularizer
# Try to answer the question: for what values of C does Gaussian kernel mimic a polynomial kernel? 
# Potentially make a plot for the above


def train_setup(X_train, Y_train, X_val, Y_val, fit_type, n_classifiers, d, kernel_type, neg = 1, pos = 2):
    

    # Encoding for one vs. all
    if fit_type == 'one_vs_all':
        Y_encoding = helpers.get_one_vs_all_encoding(Y_train, n_classifiers)
    
    # Encoding for one vs. one
    elif fit_type == 'one_vs_one': 
        Y_encoding = helpers.get_one_vs_one_encoding(Y_train, n_classifiers, neg, pos)
    
    # Get polynomial kernel
    if kernel_type == 'polynomial':
        K_train = helpers.get_polynomial_kernel(X_train, X_train, d)
        K_val = helpers.get_polynomial_kernel(X_train, X_val, d)
    
    # Get Gaussian kernel
    elif kernel_type == 'gaussian':
        K_train = helpers.get_gaussian_kernel(X_train, X_train, d)
        K_val = helpers.get_gaussian_kernel(X_train, X_val, d)

    # Store the number of samples
    n_samples = np.max(Y_train.shape)

    # Store this as a list of arrays: we don't want to repeat the lookup every time
    K_i = [K_train[i, :] for i in range(n_samples)]

    # Return statement
    return(Y_encoding, K_train, K_val, K_i, n_samples)




def train_perceptron(Y_encoding, K_train, K_val, K_i, n_samples, 
                     X_train, Y_train, 
                     X_val, Y_val, epochs, n_classifiers, 
                     question_no, convergence_epochs, fit_type, 
                     check_convergence,
                     neg=1, pos=2):
    '''
    --------------------------------------
    This is the main training loop for
    the kernel perceptron algorithm.
    '''
    # Store minimum loss for convergence check
    min_loss = np.inf
    convergence_counter = 0

    # Initialize container for weights
    # Store the number of samples
    alpha = np.zeros((n_classifiers, K_train.shape[0]))
    
    # Track mistakes here
    mistake_tracker = []

    # Run for a fixed user-specified number of epochs
    for epoch in range(epochs):

        # Initialize mistakes
        # Initialize container online predictions here
        mistakes = 0
        preds_train = []
        
        # Do this for each example in the dataset
        for i in range(n_samples):
            
            # Get predictions on the training set
            Y_hat, y_pred, signs, wrong, mistake = get_train_predictions(alpha, 
                                                                         K_i[i], 
                                                                         Y_encoding[i], 
                                                                         Y_train[i], 
                                                                         fit_type)
            # Increment the mistake counter
            # Store predictions
            mistakes += mistake
            preds_train.append(y_pred)
            
            # Update classifiers even if a single one makes a mistake
            # Enforce uniqueness in the mistake tracker
            if np.sum(wrong) > 0:
                mistake_tracker.append(i)
                mistake_tracker = list(set(mistake_tracker))
                alpha[wrong, i] -= signs[wrong]

        # Get the training prediction with the updated weights
        mistake_percent = mistakes/n_samples
        
        # Get predictions and measure training loss
        Y_hat_train, preds_train = get_final_predictions(alpha, K_train, fit_type)
        train_loss = helpers.get_loss(Y_train, preds_train)

        # Print results
        msg = 'Train loss: {}, Online mistakes: {}, Epoch: {}'
        print(msg.format(train_loss, mistake_percent, epoch))
        
        # Check convergence if user specifies to check
        if check_convergence == True:
            
            if train_loss >= min_loss:
                convergence_counter += 1
            
            else:
                convergence_counter = 0
                min_loss = train_loss

            if convergence_counter >= convergence_epochs or np.allclose(min_loss, 0):
                break

    # Get final predictions and losses
    # Still need to check if we can get rid of this
    Y_hat_train, preds_train = get_final_predictions(alpha, K_train, fit_type)
    Y_hat_val, preds_val = get_final_predictions(alpha, K_val, fit_type)

    train_loss = helpers.get_loss(Y_train, preds_train)
    val_loss = helpers.get_loss(Y_val, preds_val)

    # Store results
    if fit_type == 'one_vs_all':

        # Store a record of training and validation accuracies and other data from each epoch
        history = {
            
            "train_loss": train_loss,
            "val_loss": val_loss,
            "preds_train": preds_train,
            "preds_val": preds_val,
        }


    if fit_type == 'one_vs_one':
        
        history = {}
        history['Y_hat_train'] = Y_hat_train
        history['Y_hat_val'] = Y_hat_val
        history['preds_train'] = preds_train
        history['preds_val'] = preds_val

    # Return statement
    return(history)


def get_train_predictions(alpha, K_examples, Y_encoding, target, fit_type):
    '''
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    '''
    # Get the raw predictions for each y-value
    Y_hat = alpha @ K_examples

    # Then figure out which predictions are wrong
    # Arbitrarily assign 0 to be wrong
    wrong = (Y_encoding*Y_hat <= 0)

    # Store the sign of the predictions
    # This is used in the update
    signs = np.sign(Y_hat)
    signs[Y_hat == 0] = -1

    # Get final predictions
    # For one vs. all this is the arg. max of the raw predictions
    # We make a mistake if this does not equal the target
    if fit_type == 'one_vs_all':

        preds = np.argmax(Y_hat, axis = 0)
        mistake = (preds != target).astype(int)
        
    # For one vs. one this is the sign of prediction
    # We make a mistake if this does not equal the encoding
    elif fit_type == 'one_vs_one':
        
        preds = signs[0]
        mistake = (preds != Y_encoding).astype(int)
        
    # Return statement
    return(Y_hat, preds, signs, wrong, mistake)


def get_final_predictions(alpha, K_examples, fit_type):
    '''Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.
    '''
    
    Y_hat = alpha @ K_examples
    
    if fit_type == 'one_vs_all':
        preds = np.argmax(Y_hat, axis = 0)

    if fit_type == 'one_vs_one':
        preds = np.sign(Y_hat)
        preds[Y_hat == 0] = -1

    return(Y_hat, preds)