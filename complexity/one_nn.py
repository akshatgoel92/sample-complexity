import helpers
import numpy as np
import matplotlib.pyplot as plt




class KNN:


    def __init__(self, k, X_train, Y_train):

        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train.astype(int)
        self.distance_method = 'euclidean'
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
        val_distance = self.get_euclidean_dist(X_val, self.X_train)
        
        val_sorted_distances = np.argsort(val_distance, axis = 1)[:, :self.k]

        n_val_samples = len(Y_val)

        val_predictions = []

        # Loop through validation samples
        for i in range(n_val_samples):
        
            # Labels of nearest neighbors
            labels = self.Y_train[val_sorted_distances[i]]
        
            # Predictions based on nearest neighbor labels
            candidates, votes = np.unique(labels, return_counts = True)

            decision = np.where(votes == np.max(votes))[0].astype(int)

            candidates = candidates[decision]

            if len(votes > 1):
                val_predictions.append(np.random.choice(candidates, size=1)[0])

            else:
                val_predictions.append(candidates[0])

        self.val_loss = helpers.get_loss(Y_val, val_predictions)

        return(self.val_loss)