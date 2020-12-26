import numpy as np
import helpers


def get_dist(X, Y):
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



def train_one_nn(X_train, Y_train, X_val, Y_val):
    '''
    Train one nearest neighbor
    '''
    # Initialize empty matrix
    predictions = []

    # Calculate distance between validation point and each training point
    distance = get_dist(X_val, X_train)
    
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
            prediction.append(np.random.choice([1, -1]))
        
        else: 
            predictions.append(prediction)

    predictions = np.array(predictions)

    # Calculate validation loss
    val_loss = helpers.get_loss(Y_val, predictions)

    return(val_loss)



def get_one_nn(m, n):
    '''
    --------------------
    Run perceptron algorithm to get a base-line
    --------------------
    Parameters: 
    X: Numpy array of training features (shape = 784 X n)
    y: Binary (1/0) training label (shape = n X 1)
    --------------------
    Output: 
    w: trained weights
    y_preds: predictions
    --------------------
    '''
    # Set the random seed for np.random number generator
    # This will make sure results are reproducible
    
    
    # Prepare data for the perceptron
    X, Y = helpers.get_binary_data(m, n)

    # 
    X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, 0.8)
    
    # Call the perceptron training with the given epochs
    history = train_one_nn(X_train, Y_train, X_val, Y_val)
    
    # Return statement
    return(history)


if __name__ == '__main__':

    np.random.seed(132089)
    
    # Set parameters
    m = 100
    n = 4
    
    # Call training function
    history = get_one_nn(m, n)
    print(history)