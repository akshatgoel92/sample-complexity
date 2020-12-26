#!/usr/bin/env python
# coding: utf-8
# Import packages
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def get_data():
    '''
    ------------------------
    Input: None
    Output: Assignment data
    This function generates
    the assignment data for 
    Q 1.1
    ------------------------
    '''
    return np.array([1, 2, 3, 4]), np.array([3, 2, 0, 5])  


def get_sol(X, Y):
    '''
    ------------------------
    Input: Polynomial features
    Output: Least squares solution
    This calculates the analytical
    solution for least squares given
    X features and Y labels
    This is reused by overfitting.py
    for the overfitting section of the
    assignment
    ------------------------
    '''
    return np.linalg.solve(X.T @ X, X.T @ Y)


def get_predictions(X, beta_hat):
    '''
    ------------------------
    Input: 
           1) X: feature values for prediction points
           2) beta_hat: Least squares coefficients to use
                        for predictions
    Output: Predictions
    ------------------------
    '''
    return X @ beta_hat


def run_regression(k, x, y):
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
    phi_x = get_polynomial_basis(x, k)
    beta_hat = get_sol(phi_x, y)
    y_hat = get_predictions(phi_x, beta_hat)
    
    mse = get_mse(y, y_hat)
    ln_mse = get_ln_mse(mse)
    
    results = {'beta_hat': beta_hat, 'y_hat': y_hat, 
               'mse': mse, 'ln_mse': ln_mse, 
               'degree': k-1, 'dim': k}
    
    return(results)


def get_final_results(results):
    '''
    ------------------------
    Input: Results dictionary
    Output: Dataframe with two columns:
    Degree and MSE
    This is a convenience function
    to display results in an easily 
    readable way.
    ------------------------
    '''
    mse = pd.DataFrame([result['mse'] for result in results], columns = ['MSE'])
    mse['degree'] = [result['degree'] for result in results]
    mse.set_index('degree', inplace = True)
    
    return(mse)



if __name__ == '__main__':
    main()