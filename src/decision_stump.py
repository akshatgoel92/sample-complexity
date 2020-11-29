# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers

def get_split(X, i, threshold):
    '''
    This functions takes in 
    a dataset as a numpy array 
    and a pixel value i to split 
    on as well as a threshold value
    to use and returns split datasets
    based on whether or not each row in
    the original data is greater than the
    threshold.
    '''
    below_threshold = (X[:, i] < threshold)
    return(X[below_threshold], X[~below_threshold])




def get_gini_coefficient(groups):
    '''
    Calculates the Gini coefficient
    given a set of groups which are
    splits of the training data.
    '''
    gini = 0
    n_instances = float(sum([len(group) for group in groups]))

    for group in groups:
        
        size = len(group)
        if size == 0: continue
        
        classes = np.unique(Y, return)
        p = np.unique(group, return_counts=True)/size
        score += p**2 
        gini += (1.0 - score) * (size / n_instances)

    return(gini)




if __name__ == '__main__':

    path = os.path.join('..', 'data')
    name = 'zipcombo.dat'
    X, Y = helpers.load_data(path, name)


