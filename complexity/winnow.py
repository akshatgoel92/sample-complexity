# Import packages
import helpers
import numpy as np


class Winnow():


    def __init__(self, X_train, Y_train, epochs, n):

        self.epochs = epochs
        self.X_train = X_train
        self.Y_train = Y_train
        self.train_loss = -1
        self.val_loss = -1
        self.w = np.ones(n)
        self.n = n

        self.X_train[self.X_train == -1] = 0
        self.Y_train[self.Y_train == -1] = 0

    
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

        # Run for a fixed number of epochs
        for epoch in range(self.epochs):

            mistakes=0
            # Do this for each example in the dataset
            for i in range(self.X_train.shape[0]):

                # Store the sample data
                x_i = self.X_train[i, :]
                y_i = self.Y_train[i]

                # Compute the prediction with the current weights
                if (np.dot(w, x_i) >= self.n): y_hat = 1
                else: y_hat = 0
            
                # Check if the prediction is correct against the labels
                # If it is correct we don't need to make any updates: we just move to the next iteration
                # If it is not correct then we do the following: 
                # 1) Update the weights and biases in the direction of the label
                if y_hat != y_i:
                    w = w*np.power(2.0, (y_i - y_hat)*x_i)

        y_train_preds = ((np.dot(self.X_train, w)) >= self.n).astype(int)
        
        self.train_loss = helpers.get_loss(self.Y_train, y_train_preds)
        self.w = w

        return(self.train_loss)        
        

    def validate(self, X_val, Y_val):
        '''
        Validate perceptron
        '''
        w = self.w

        X_val[X_val == -1] = 0
        Y_val[Y_val == -1] = 0
        
        y_val_preds = ((np.dot(X_val, w)) >= self.n).astype(int)
        self.val_loss = helpers.get_loss(Y_val, y_val_preds)

        return(self.val_loss)
            


if __name__ == '__main__':

    np.random.seed(132089)
    
    # Set parameters
    epochs = 100
    m = 20
    n = 22
    m_sample_test = 1000

    X_train, Y_train = helpers.get_binary_data(m, n)
    X_val, Y_val = helpers.get_binary_data(m_sample_test, n)
    
    # Call training function
    winnow = Winnow(X_train, Y_train, epochs, n)
    train_loss = winnow.fit()
    val_loss = winnow.validate(X_val, Y_val)