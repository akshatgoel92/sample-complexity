# Import packages
import os
import helpers
import numpy as np
import matplotlib.pyplot as plt


class Net():
    
    
    def __init__(self, layers, momentum, lr, epochs, X, Y, n_classes):
        '''
        Initialize neural network classifier weights
        to the given architecture
        '''
        # Initialize class variables
        self.X = X
        self.Y = Y
        self.layers = layers
        self.input_dim = X.shape[1]
        self.output_dim = n_classes
        
        # Optimizer parameters
        self.momentum = momentum
        self.epochs = epochs
        self.lr = lr
        self.Z = []
        self.A = []
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        
        # Loop through the layers to initialize them
        for i, layer in enumerate(layers):
            # The input layer needs to use the input dimension shape
            if i == 0:
                self.W.append(np.random.randn(layer, self.input_dim))
                self.b.append(np.random.randn(layer, 1))
            # The hidden layers use the dimension of the layer directly previous
            else:
                self.W.append(np.random.randn(layer, layers[i-1]))
                self.b.append(np.random.randn(layer, 1))
        
        # Append the output layer
        self.W.append(np.random.randn(self.output_dim, layers[-1]))
        self.b.append(np.random.randn(self.output_dim, 1))


    def forward(self):
        '''
        Propagate the weights forward
        through the network
        '''
        # Create first inputs
        z = self.W[0] @ self.X.T + self.b[0]
        a = helpers.sigmoid(z)

        # Create containers to iterate over
        Z = [z]
        A = [a]

        # Now iterate through the rest of the layers
        for i in range(1, len(layers)):
            
            print(self.W[i].shape)
            print(A[i-1].shape)
            
            z = self.W[i] @ A[i-1] + self.b[i]
            a = helpers.sigmoid(z)
            
            Z.append(z)
            A.append(a)

        # Make predictions 
        Z.append(self.W[-1] @ A[-1] + self.b[-1])
        A.append(helpers.sigmoid(Z[-1]))

        # Pass final layer through softmax to get predictions
        A[-1] = np.argmax(A[-1], axis = 0)
        
        # Return statement
        return(Z, A)


    def backward(self):
        
        # Initialize derivatives
        dA = [self.A[-1] - self.Y]/m
        dZ = []
        dW = []
        db = []

        # Go through the reversed list
        for i in reversed(range(1, len(layers))):
            
            # Append the relevant derivative
            dZ.append(dA[i] @ helpers.sigmoid_derivative(self.Z[i]))
            dW.append(dZ[i] @ self.A[i-1])
            db.append(np.sum(dZ[i]))

        # Now append the last derivative dW0  
        dW.append(dZ[0] @ self.X)
        db.append(np.sum(dZ[0]))

        return(reversed(dW), reversed(db))


    def update(self, dW, db):
        '''
        Make gradient descent update
        '''
        self.W = self.W - self.lr*self.dW
        self.b = self.b - self.lr*self.db


def train(self):
    '''
    Execute the training steps above and generate
    the results that have been specified in the report.
    '''
    # Loop through each epoch
    for epoch in range(self.epochs):

        # Call the perceptron training with the given epochs
        self.Z, self.A = forward(self)
        self.dW, self.db = backward(self)
        self.W, self.b = update(dW, db)
        



if __name__ == '__main__':
    
    # Set data paths
    data_path = os.path.join('..', 'data')
    name = 'zipcombo.dat'

    # Load and split
    X, Y = helpers.load_data(data_path, name)

    # Set training and testing dataset
    train_percent = 0.8
    X, Y = helpers.shuffle_data(X, Y)
    X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, train_percent)

    # Set parameters
    layers = [64, 32, 16]
    momentum = 0.9
    n_classes=10
    epochs = 10
    lr = 0.009

    # Create object
    net = Net(layers, epochs, lr, momentum, X_train, Y_train, n_classes)
    Z, A = net.forward()


