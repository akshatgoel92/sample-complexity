import helpers
import numpy as np
import matplotlib.pyplot as plt




class KNN:


    def __init__(self, k, X_train, Y_train, X_val, Y_val):

        self.k = k
        self.X_train = X_train
        self.Y_train = Y_train.astype(int)
        self.X_val = X_val
        self.Y_val = Y_val
        self.distance_method = 'euclidean'
        
        # Initialize empty matrix to store predictions
        # Store no. of validation samples
        self.predictions = []
        self.n_val_samples = len(self.Y_val)


    def get_euclidean_dist(self, X, Y):
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
        return(dist)



    def train(self):
        '''
        Train one nearest neighbor
        '''
        # Calculate distance between validation point and each training point
        distance = self.get_euclidean_dist(self.X_val, self.X_train)
        
        sorted_distances = np.argsort(distance, axis = 1)[:, :self.k]

        # Loop through validation samples
        for i in range(self.n_val_samples):
        
            # Labels of nearest neighbors
            self.labels = self.Y_train[sorted_distances[i]]
        
            # Predictions based on nearest neighbor labels
            self.candidates, self.votes = np.unique(self.labels, return_counts = True)

            print(self.candidates, self.votes)

            decision = np.where(self.votes == np.max(self.votes))[0].astype(int)

            self.candidates = self.candidates[decision]

            
            if len(self.votes > 1):

                self.predictions.append(np.random.choice(self.candidates, size=1)[0])

            else:
                self.predictions.append(self.candidates[0])

        # Calculate validation loss
        val_loss = helpers.get_loss(self.Y_val, self.predictions)

        return(val_loss)


if __name__ == '__main__':

    np.random.seed(13290)

    data_args = {

        'data_path': '../data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5,

    }

    # Load full dataset
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
    
    # Shuffle and split dataset
    X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X,Y)

    k = 20
    
    # Split dataset
    X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

    # Call the perceptron training with the given epochs
    knn = KNN(k, X_train, Y_train, X_val, Y_val)

    val_loss = knn.train()

    print(val_loss)