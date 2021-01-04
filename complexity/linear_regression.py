# Import packages
import os
import numpy as np
import helpers

class LinearRegression():


    def __init__(self, X_train, Y_train):

        self.beta_hat = 0
        self.X_train = X_train
        self.Y_train = Y_train
        self.train_loss = -1
        self.val_loss = -1



    def fit(self):
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
        self.beta_hat = np.linalg.pinv(self.X_train) @ self.Y_train
        self.y_hat_train = np.sign(self.X_train @ self.beta_hat)
        self.y_hat_train[self.y_hat_train == 0] = -1
    
        self.train_loss = helpers.get_loss(self.Y_train, self.y_hat_train)
        
    
        return(self.train_loss)


    def validate(self, X_val, Y_val):
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
        y_hat_val = np.sign(X_val @ self.beta_hat)
        y_hat_val[y_hat_val == 0] = -1
        
        self.val_loss = helpers.get_loss(Y_val, y_hat_val)

        return(self.val_loss)