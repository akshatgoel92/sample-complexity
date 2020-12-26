import numpy as np


def get_pairwise_dist(A, B):
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
    dist = np.einsum('ij,ij->i',A, A)[:,None] + np.einsum('ij,ij->i',B,B) - 2*np.dot(A,B.T)
    return(dist)



def one_nearest_neighbor(X_train, Y_train, X_val, Y_val, epochs, lr):

    predictions = []

    for i in range(X_val.shape[0]):

        distance = get_euclidean_distance(X_val[i, :], X_train)
        neighbors = np.argmin(distance)
        labels = Y_train[neighbors]
        prediction = np.sign(np.sum(labels))
        
        if prediction == 0: prediction.append(np.random.choice([1, -1]))
        else: predictions.append(prediction)

    return(np.array(prediction))