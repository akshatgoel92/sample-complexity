import numpy as np

'''
Input: (x1,y1),...,(xm,ym)∈{0,1}n ×{0,1} ⃗
1. Initialise w1 = 1.
2. For t=1 to m do
3. Receive pattern: xt ∈ {0, 1}^n 
4. Predict: 
    0 if wt·xt<n
yˆ = t
5. Receive label: yt ∈ {0, 1} 
6. If mistake (yˆ ̸= y )
• Update: wt+1,i = wt,i 2(yt −yˆt ) xt,i (1 ≤ i ≤ n). 
7. End For
'''

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





# Import packages
import helpers
import numpy as np



def train_perceptron(X, Y, X_val, Y_val, epochs, lr):
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
    w = np.zeros(X.shape[1])
    b = 0
    
    # History goes here
    history = {
        "weights": [w],
        "losses": [], 
        "biases": [b],
        "train_loss": [],
        "val_loss": []
    }
    
    
    # Run for a fixed number of epochs
    for epoch in range(epochs):
        
        mistakes=0

        # Do this for each example in the dataset
        for i in range(X.shape[0]):

            # Store the sample data
            x_i = X[i, :]
            y_i = Y[i]

            
            # Compute the prediction with the current weights
            if (np.dot(w, x_i) > 0): y_hat = 1
            else: y_hat = -1
            
            # Check if the prediction is correct against the labels
            # If it is correct we don't need to make any updates: we just move to the next iteration
            # If it is not correct then we do the following: 
            # 1) Update the weights and biases in the direction of the label
            if y_hat != y_i:
                mistakes+=1
                print(mistakes)
                w += (y_i - y_hat)*x_i
                b += (y_i - y_hat)
                
            
            
        # Get predictions on train and test
        y_train_preds = np.sign(np.dot(X, w) + b)
        y_train_preds[y_train_preds==0] = -1
        
        y_val_preds = np.sign(np.dot(X_val, w) + b)
        y_val_preds[y_val_preds==0] = -1

        # Training accuracy
        train_loss = helpers.get_loss(Y, y_train_preds)
        val_loss = helpers.get_loss(Y_val, y_val_preds)
        print("Epoch {}/{}: Training loss = {}, Val. loss = {}".format(epoch, epochs, train_loss, val_loss))
         
        # Append results to history
        history["biases"].append(b)
        history["weights"].append(w)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # Get training accuracy
        print("Epoch {}/{}: Training Loss = {}".format(epoch, epochs, train_loss))
    
    # Return statement
    return(history)


def get_perceptron_baseline(m, n, epochs, lr):
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
    history = train_perceptron(X_train, Y_train, X_val, Y_val, epochs, lr)
    
    # Return statement
    return(history)




if __name__ == '__main__':

    np.random.seed(132089)
    
    # Set parameters
    epochs = 20
    m = 10000
    lr = 1
    n = 30
    
    # Call training function
    history = get_perceptron_baseline(m, n, epochs, lr)