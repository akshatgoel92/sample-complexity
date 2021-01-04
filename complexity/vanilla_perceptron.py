# Import packages
import helpers
import numpy as np


class Perceptron():


    def __init__(self, X_train, Y_train, epochs):

        self.epochs = epochs
        self.X_train = X_train
        self.Y_train = Y_train
        self.train_loss = -1
        self.val_loss = -1
        self.w = np.zeros(self.X_train.shape[1])
        self.b = 0



    def fit(self):
        '''
        --------------------
        Perceptron algorithm
        --------------------
        Parameters: 
        X: Numpy array of training features (shape = 784 X n)
        y: Binary (1/-1) training label (shape = n X 1)
        --------------------
        Output: 
        w: trained weights
        b: trained biases
        y_preds: predictions 
        --------------------
        '''
        # Initialize weights and biases
        w = self.w
        b = self.b
    
    
        # Run for a fixed number of epochs
        for epoch in range(self.epochs):
        
            mistakes=0
            
            # Do this for each example in the dataset
            for i in range(self.X_train.shape[0]):

                # Store the sample data
                x_i = self.X_train[i, :]
                y_i = self.Y_train[i]

            
                # Compute the prediction with the current weights
                if (np.dot(w, x_i) + b > 0): y_hat = 1
                else: y_hat = -1
            
                # Check if the prediction is correct against the labels
                # If it is correct we don't need to make any updates: we just move to the next iteration
                # If it is not correct then we do the following: 
                # 1) Update the weights and biases in the direction of the label
                if y_hat != y_i:
                    mistakes+=1
                    w += (y_i - y_hat)*x_i
                    b += (y_i - y_hat)
                
        # Get predictions on train and test
        y_train_preds = np.sign(np.dot(self.X_train, w) + b)
        y_train_preds[y_train_preds==0] = -1

        self.w = w
        self.b = b
        self.train_loss = helpers.get_loss(self.Y_train, y_train_preds)

        return(self.train_loss)


    def validate(self, X_val, Y_val):
        '''
        Validate perceptron
        '''
        y_val_preds = np.sign(np.dot(X_val, self.w) + self.b)
        y_val_preds[y_val_preds==0] = -1

        # Training accuracy
        self.val_loss = helpers.get_loss(Y_val, y_val_preds)

        return(self.val_loss)