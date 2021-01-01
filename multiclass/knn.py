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
        self.train_predictions = []
        self.val_predictions = []
        self.n_train_samples = len(self.Y_train)
        self.n_val_samples = len(self.Y_val)


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


    def train(self):

        
        train_distance = self.get_euclidean_dist(self.X_train, self.X_train, fill=True)

        train_sorted_distances = np.argsort(train_distance, axis = 1)[:, :self.k + 1]

        # Loop through train samples
        for j in range(self.n_train_samples):
        
            # Labels of nearest neighbors
            sorted_distance = train_sorted_distances[j]

            sorted_distance = sorted_distance[sorted_distance != j]

            print(sorted_distance)

            labels = self.Y_train[sorted_distance]

            print(len(labels))
        
            # Predictions based on nearest neighbor labels
            candidates, votes = np.unique(labels, return_counts = True)
            
            # Print candidates and votes as a check during testing
            print("Votes:")
            print(candidates, votes)

            # Make a decision by majority vote
            decision = np.where(votes == np.max(votes))[0].astype(int)
            candidates = candidates[decision]

            # Print the final decision as a check by majority vote
            print("Decision:")
            print(decision)
            print(candidates)

            # Break ties if there is more than one maximum vote
            if len(votes > 1):
                self.train_predictions.append(np.random.choice(candidates, size=1)[0])
            # Otherwise just append the single final decision to the predictions vector 
            else:
                self.train_predictions.append(candidates[0])



    def validate(self):
        '''
        Train one nearest neighbor
        '''
        # Calculate distance between validation point and each training point
        val_distance = self.get_euclidean_dist(self.X_val, self.X_train)
        
        val_sorted_distances = np.argsort(val_distance, axis = 1)[:, :self.k]

        # Loop through validation samples
        for i in range(self.n_val_samples):
        
            # Labels of nearest neighbors
            labels = Y_train[val_sorted_distances[i]]
        
            # Predictions based on nearest neighbor labels
            candidates, votes = np.unique(labels, return_counts = True)
            print("Votes:")
            print(candidates, votes)

            decision = np.where(votes == np.max(votes))[0].astype(int)

            print("Decision:")
            print(decision)

            candidates = candidates[decision]

            print(candidates)

            if len(votes > 1):
                self.val_predictions.append(np.random.choice(candidates, size=1)[0])

            else:
                self.val_predictions.append(candidates[0])

        # Calculate validation loss
        train_loss = helpers.get_loss(self.Y_train, self.train_predictions)
        val_loss = helpers.get_loss(self.Y_val, self.val_predictions)

        return(train_loss, val_loss)


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

    k = 2
    
    # Split dataset
    X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

    # Call the perceptron training with the given epochs
    knn = KNN(k, X_train, Y_train, X_val, Y_val)

    train_loss = knn.train()
    val_loss = knn.validate()

    print(train_loss, val_loss)