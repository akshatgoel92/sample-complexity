import numpy as np
import helpers





class OneNN():


    def __init__(self, X_train, Y_train):

        self.predictions = []
        self.X_train = X_train
        self.Y_train = Y_train
        self.val_loss = -1


    def get_euclidean_dist(self, X, Y, fill=False):
        '''
        --------------------------
        Load data from source
        Input: A: An array of x-values np.array
               B: Another array of x-values np.array
        Output: Pairwise Euclidean distances between A and B
    
        The output from this function is used 
        to calculate the Gaussian kernel. We used
        einsum here because we read that it can 
        provide a speed advantage in these 
        situations.
    
        References for einsum:
        https://ajcr.net/Basic-guide-to-einsum/
        -------------------------
        '''
        # 1) Element wise product and then sum horizontally
        # 1) Element wise product and then sum horizontally
        # 2) Dot product between A and B.T
        dist = np.sqrt(np.einsum('ij,ij->i',X, X)[:,None] + np.einsum('ij,ij->i',Y,Y) - 2*np.dot(X,Y.T))

        if fill == True:
            np.fill_diagonal(dist, 0)

        return(dist)



    def validate(self, X_val, Y_val):
        '''
        Train one nearest neighbor
        '''
        # Calculate distance between validation point and each training point
        distance = self.get_euclidean_dist(X_val, X_train)
    
        # Store no. of validation samples
        n_val_samples = X_val.shape[0]

        # Loop through validation samples
        for i in range(n_val_samples):
        
            # Labels of nearest neighbors
            labels = Y_train[np.where(distance[i] == np.min(distance[i]))]
        
            # Predictions based on nearest neighbor labels
            prediction = np.sign(np.sum(labels))
        
            # Add prediction to sequence
            if prediction == 0: 
                self.predictions.append(np.random.choice([1, -1]))
            else: 
                self.predictions.append(prediction)

        # Wrap predictions in an array
        self.predictions = np.array(self.predictions)

        # Calculate validation loss
        self.val_loss = helpers.get_loss(Y_val, self.predictions)

        return(self.val_loss)



if __name__ == '__main__':
    m = 90
    n = 8
    
    X, Y = helpers.get_binary_data(m, n)
    X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, 0.8)

    one_nn = OneNN(X_train, Y_train)
    val_loss = one_nn.validate(X_val, Y_val)

    print(val_loss)