import helpers
import numpy as np
import matplotlib.pyplot as plt


def get_initial_weights(n_classes, n_features):
    '''
    Initialize the weight and bias vectors
    '''
    W = np.zeros((n_classes, n_features + 1))
    
    return(W)


def get_one_hot_encoding(Y_train, n_classes):
    '''
    --------------------------------------
    Get one hot encoded labels for 1 vs. all
    '''
    Y = np.full(Y_train.size*n_classes, 0).reshape(Y_train.size, n_classes)
    Y[np.arange(Y_train.size), Y_train] = 1

    return(Y)


def add_bias(X):
    '''
    Takes in n_examples * n_features
    Returns (n_examples * n_features + 1) matrix
    '''
    return(np.hstack((np.ones((max(X.shape),1)),X)))



def get_cost(Y_encoding, a, n_examples):
    '''
    Returns cross entropy loss
    '''

    return -np.sum(Y_encoding * np.log(a)) / n_examples



def get_softmax(a):
    '''
    Takes in a vector of probabilities and 
    outputs the thresholded predictions
    '''
    return np.exp(a) / np.sum(np.exp(a), axis=1).reshape(a.shape[0], 1)


def predict_train(X, W):
    '''
    Generate the prediction
    '''
    a = get_softmax(np.dot(X, W.T))

    return(a)


def get_gradient_descent_step(lr, a, X, Y, n_examples, W):
    '''
    Update weights according to gradient descent rule
    '''
    delta = (lr/n_examples) * np.dot((a-Y).T, X)
    W -= delta
    
    return(W)


def train(X_train, Y_train, lr = 0.01, n_classes = 10, epochs = 100):
    '''
    Update weights according to gradient descent rule
    '''
    history = []
    
    n_features = min(X_train.shape)

    n_examples = len(Y_train)

    X_train = add_bias(X_train)

    Y_train = Y_train.astype(int)    

    Y_encoding = get_one_hot_encoding(Y_train, n_classes)
    
    W = get_initial_weights(n_classes, n_features)

    for epoch in range(epochs):

        a = predict_train(X_train, W)

        cost = get_cost(Y_encoding, a, n_examples)

        W = get_gradient_descent_step(lr, a, X_train, Y_encoding, n_examples, W) 

        history.append(cost)

    return(W, history)


def predict(X_val, Y_val, W):
    '''
    Predict on validation set
    '''

    X_val = add_bias(X_val)

    Y_val = Y_val.astype(int)

    n_val_examples = len(Y_val)

    preds_val = predict_train(X_val, W)

    preds_val = np.argmax(preds_val, axis = 1)

    val_loss = helpers.get_loss(Y_val, preds_val)

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
    
    # Split dataset
    X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

    # Train
    W, history = train(X_train, Y_train)

    # Evaluate
    val_loss = predict(X_val, Y_val, W)

