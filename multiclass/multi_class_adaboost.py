# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_weak_learner_predictions():
    '''
    Initialize boosting weights
    '''
    pass


def boost_init(m):
    '''
    Initialize boosting weights
    '''
    D = np.array([1/m]**m)
    return(D)


def boost_choose_alpha(D, preds, labels):
    '''
    Initialize boosting weights
    '''
    error = np.dot(D, np.sum(preds != labels)/len(labels))
    alpha = 0.5*np.log((1 - error/error))
    return(alpha)


def boost_update(D, alpha, labels, preds):
    '''
    Update the distribution over the training examples
    '''
    D = np.dot(D, np.exp(-1*np.dot(alpha, np.dot(labels, preds.T))))
    Z = np.sum(D)
    D = D/Z
    return(D, Z)


def boost_train(X, Y):
    '''
    Update the distribution over the training examples
    '''
    pass


def main():
    pass


if __name__ '__main__':
    main()