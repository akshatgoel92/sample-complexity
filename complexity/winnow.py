import numpy as np


def winnow(set, weights, theta, learningRate=1):

    # Initialize error count
    errors = 0
    
    # Initiate the feature vector
    wt = array([0] * len(weights))
    
    # For each entry t in the training data set
    for xt, yt in set:
        
        # Convert label to the winnow specification
        yt = -1 if yt == 0 else 1
        
        # Set the respective values to the feature vector
        wt *= 0
        for word in xt:
            wt[dictionary[word]] = 1
        
        # Calculate the dot product
        wx = dot(weights, wt)
        
        # If error then update weight vector
        if wx < theta and yt == 1:
            errors += 1
            weights += 2.0 * wt
        if wx > theta and yt == -1:
            errors += 1
            weights += 0.5 * wt
            
    # Returns the results
    return [weights, errors]