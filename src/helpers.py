# Import packages
import os
import pickle
import numpy as np
import pandas as pd 
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform


def load_data(path, name):
    '''
    --------------------------------
    Takes in a folder path
    and dataset name and loads
    the corresponding dataset
    divided into features X
    and labels Y as numpy arrays.
    --------------------------------
    '''
    data = np.loadtxt(os.path.join(path, name))
    X, Y = data[:, 1:], data[:, 0]

    return(X, Y)


def shuffle_data(X, Y):
    '''
    Takes in datasets X and Y
    and returns a random permutation
    of these datasets.
    '''
    perm = np.random.permutation(max(Y.shape))
    X = X[perm, :]
    Y = Y[perm]

    return(X, Y)


def show_images(arr, shape=(16, 16)):
    '''
    Takes in a numpy array
    of pixel values and an image shape 
    and displays the associated image.
    '''
    img = arr.reshape(shape)
    imgplot = plt.imshow(img)


def split_data(X, Y, train_percent):
    '''
    Take datasets X and Y and split them 
    into train_percent*100 % training dataset
    and (1 - train_percent)*100 % hold-out dataset.
    '''
    # Calculate no. of training examples based on user specified percentage
    # Here we use 2/3, 1/3 by default as required by the assignment
    n_train = round(train_percent*max(Y.shape))
    
    # Filter the dataframe to get training and testing rows
    X_train = X[:n_train, :]
    Y_train = Y[:n_train]
    
    # Validation set
    X_val = X[n_train:, :]
    Y_val = Y[n_train:]
    
    # Return statement
    return(X_train, X_val, Y_train, Y_val)


def get_polynomial_kernel(X, X_, d):
    '''
    Take in two matrices X and X_ and
    a kernel parameter d and return the
    Gram Matrix K(x, x_) for polynomial 
    kernel with parameter d. Note that 
    this will return polynomials of exactly
    degree d like the assignment says and 
    not a sum of polynomials upto degree d.
    '''
    return(np.power(np.dot(X, X_.T), d))


def get_gaussian_kernel(X, X_test, c):
    '''
    --------------------------
    Input: X: Training matrix
           X_test: Testing matrix
           sigma: Parameter for Kernel
    Output: Gaussian kernel matrix K(X, X_test)
    
    This function computes the 
    Gaussian kernel matrix for X and X_test.
    It calls the pairwise distance function above
    to first create the matrix of distances. Then 
    It scales and exponentiates them to recover 
    the kernel values.
    -------------------------
    '''
    K = np.einsum('ij,ij->i',X, X)[:,None] + np.einsum('ij,ij->i',X_test,X_test) - 2*np.dot(X,X_test.T)
    K = np.exp(K*-c)
    return(K)


def get_k_folds(X, Y, k):
    '''
    Take in two arrays for features
    and corresponding labels respectively
    as well as a user-specified number of
    folds. Return the X and Y arrays divided
    into the k sub-arrays where each sub-array
    is a fold.
    '''
    X_folds = np.array_split(X.copy(), k)
    Y_folds = np.array_split(Y.copy(), k)
    
    return(X_folds, Y_folds)


def get_accuracy(target, pred):
    '''
    Returns binary accuracy given 
    two arrays containing true values
    and predicted values respectively.
    '''
    return np.sum(target==pred)/max(target.shape)


def get_confusion_matrix(target, pred):
    '''
    Returns a confusion matrix given two
    arrays containing the true values 'target'
    and predicted values 'pred' respectively.
    Interpretation: we put target values in the 
    rows and predicted values in the columns. So 
    we have that for example the element (2, 1) 
    contains all the elements which have true labels
    2 but are classified as 1.
    '''
    # The confusion matrix should be a square matrix with
    # no. of rows and columns equal to the number of unique
    # values in the target: this is what we compute below

    # Compute the no. of unique values in target
    cf_dim = len(np.unique(target))
    
    # Initialize the confusion matrix as zeros
    cf= np.zeros((cf_dim, cf_dim))

    # Now go through each target value
    # Look at the corresponding prediction
    # Update the corresponding cell in the confusion matrix
    for i in range(len(target)):
        cf[int(target[i]) - 1, int(pred[i]) - 1] += 1
    
    # Return statement
    return(cf)


def get_loss_plot(results, lab, run_no, param):
  '''
  Convenience function to plot results
  '''
  # Store model name and destination path
  model_name = str(run_no) + '_' + str(param) + '_results.png'
  path = os.path.join('figs', model_name)
  
  # Plot the results
  plt.plot(results['train_' + lab], label='Train')
  plt.plot(results['val_' + lab], label='Validation')
  
  # Add annotations
  plt.legend()
  plt.title(lab.title() + ' by Epoch')
  
  # Save the figure and close the plot
  plt.savefig(path)
  plt.clf()


def get_best_results(history):
    '''
    This function takes in a 
    dictionary containing an
    epoch-wise record of training
    and validation set binary accuracies
    and returns the epoch at which the highest
    binary accuracy was reached on the dev. set 
    along with the associated accuracies on both
    training and dev. set at that epoch.
    '''
    # Store results
    best_epoch = np.array(history["val_accuracies"]).argmax()
    best_training_accuracy = history['train_accuracies'][best_epoch]
    best_dev_accuracy = history['val_accuracies'][best_epoch]

    return(best_epoch, best_training_accuracy, best_dev_accuracy)



def get_one_vs_all_encoding(Y_train, n_classes):
    '''
    --------------------------------------
    Get one hot encoded labels for 1 vs. all
    --------------------------------------
    '''
    Y = np.full(Y_train.size*n_classes, -1).reshape(Y_train.size, n_classes)
    Y[np.arange(Y_train.size), Y_train] = 1

    return(Y)


def save_results(results, question_no):
    '''
    Save results according to question no.
    '''
    with open(os.path.join('results', '{}_results.txt'.format(question_no)), 'wb') as f:
        pickle.dump(results, f)



def save_experiment_results(results, question_no):

    train_acc = []
    test_acc = []
    train_std = []
    test_std = []
    params = []

    for result in results:
        params.append(result['params'])
        train_acc.append(np.mean(np.array(result['best_training_accuracy'])))
        test_acc.append(np.mean(np.array(result['best_dev_accuracy'])))
        train_std.append(np.std(np.array(result['best_training_accuracy'])))
        test_std.append(np.std(np.array(result['best_dev_accuracy'])))

    
    results_df = pd.DataFrame([train_acc, train_std, test_acc, test_std], columns = [params], 
                        index = ['Train Loss', 'Train SD', 'Test Loss',  'Test SD'])

    results_df.to_csv(os.path.join("results", "table_{}.csv".format(question_no)))

    # Print out results each time
    print(results_df)

    return(results_df)


def open_results(question_no):
    '''
    Save results according to question no.
    '''
    with open(os.path.join('results', '{}_results.txt'.format(question_no)), 'rb') as f:
        results = pickle.load(f)

    return(results)


def compute_final_cf(n_classes):

    results = open_results('3_2')

    n_cf = len(results)
    cf = np.zeros((n_classes, n_classes))

    for result in results:
        cf = np.add(cf, result['val_cf'])

    return(cf)


def get_cv_results(histories, k = 5):
    '''
    --------------------------------
    This function takes in a 
    list of history dictionaries
    where the list is of length k. 
    k is the no. of folds we have
    used in cross-validation. This
    will return the mean value at
    every epoch across folds.
    --------------------------------
    '''
    avg_history = {}
    avg_history['train_accuracies'] = np.mean(np.array([history['train_accuracies'] for history in histories]), axis = 0)
    avg_history['val_accuracies'] = np.mean(np.array([history['val_accuracies'] for history in histories]), axis = 0)

    return(avg_history)


def sigmoid(x):
    '''
    --------------------------------
    Calculates sigmoid activation value at x.
    --------------------------------
    '''
    return 1/(1+np.exp(-x))


def sigmoid_derivative():
    pass