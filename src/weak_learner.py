# Import packages
import numpy as np
import helpers
import matplotlib.pyplot as plt


class NeuralNet(self):
    
    
    def __init__(self, layers, lr, momentum, epochs, X, Y):
        '''
        Initialize neural network classifier weights
        to the given architecture
        '''
        # Initialize class variables
        self.X = X
        self.Y = Y
        self.layers = layers
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        # Optimizer parameters
        self.lr = lr
        self.momentum = momentum
        
        # How many passes to make thhrough the data
        self.epochs = epochs
        
        # Initialize containers for weights and biases
        self.W = []
        self.b = []
        
        # Loop through the layers to initialize them
        for i, layer in enumerate(layers):
            # The input layer needs to use the input dimention shape
            if i == 0:
                self.W.append(np.random.randn((layer, input_dim)))
                self.b.append(np.random.randn((layer, 1)))
            # The hidden layers use the dimension of the layer directly previous
            else:
                self.W.append(np.random.randn((layer, layers[i-1]))
                self.b.append(np.random.randn((layer, 1)))
        # Append the output layer
        self.W.append(np.random.randn(output_dim, layers[-1])))


    def forward(self):
        '''
        Propagate the weights forward
        through the network
        '''
        # Create first inputs
        z = self.W[0] @ self.X + self.b[0]
        a = helpers.sigmoid(z)

        # Create containers to iterate over
        Z = [z]
        A = [a]


        # Now iterate through the rest of the layers
        for i in range(1, len(layers)):
            
            z = self.W[i]@A[i-1] + self.b[i]
            a = helpers.sigmoid(z)
            
            Z.append(z)
            A.append(a)


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

        return(reversed(dL), reversed(dZ), reversed(dA), reversed(dW), reversed(db))


    def update(self):
        '''
        Make gradient descent update
        '''
        self.W = self.W - self.lr*self.dW
        self.b = self.b - self.lr*self.db


    def train():
        pass




if __name__ == '__main__':
    
    layers = []


