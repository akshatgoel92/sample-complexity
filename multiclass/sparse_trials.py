def get_sparse_predictions(alpha, K_examples, non_zero_alphas):
    '''
    Use only non-zero elements to make predictions
    '''
    Y_hat = []

    for classifier in range(alpha.shape[0]):
        Y_hat.append(alpha[classifier][non_zero_alphas[classifier]] \
                    @ K_examples[non_zero_alphas[classifier]])

    Y_hat = np.array(Y_hat)
    preds = np.argmax(Y_hat, axis = 0)
    
    return(Y_hat, preds)


def get_sparse_update(Y_train, Y_hat, alpha, n_classes, i, Y):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.

    # Now first make a matrix Y with dim(Y) ---> (n_classes,) which is only filled with -1
    # Then get the label from the Y_train matrix
    # If this label is 6 then we want to change the 6th index to 1
    # Y = get_one_hot_encoding(n_classes, Y_train, i)
    # Compute sign of predictions and store indices to update        
    # Check if the prediction is correct against the labels
    # If it is correct we don't need to make any updates: we just move to the next iteration
    # If it is not correct then we update the weights and biases in the direction of the label
    --------------------------------------
    '''
    Y = Y[i]
    signs = np.ones(Y_hat.shape)
    signs[Y_hat <= 0] = -1
    signs[Y == 0] = 0
    wrong = (Y*Y_hat <= 0).astype(int)

    # Make the update if any of the classifiers are wrong
    if np.sum(wrong) > 0:
        alpha = alpha.tolil()
        alpha[:, i] -= (signs*wrong).reshape(alpha[:, i].shape)
        alpha = alpha.tocsc()
        
    return(alpha)


def get_sparse_update_2(Y_train, Y_hat, alpha, n_classes, i, Y, non_zero_alphas):
    '''
    --------------------------------------
    Returns raw predictions and class predictions
    given alpha weights and Gram matrix K_examples.

    # Now first make a matrix Y with dim(Y) ---> (n_classes,) which is only filled with -1
    # Then get the label from the Y_train matrix
    # If this label is 6 then we want to change the 6th index to 1
    # Y = get_one_hot_encoding(n_classes, Y_train, i)
    # Compute sign of predictions and store indices to update        
    # Check if the prediction is correct against the labels
    # If it is correct we don't need to make any updates: we just move to the next iteration
    # If it is not correct then we update the weights and biases in the direction of the label
    --------------------------------------
    '''
    Y = Y[i]
    signs = np.ones(Y_hat.shape)
    signs[Y_hat <= 0] = -1
    wrong = (Y*Y_hat <= 0)
    
    if np.sum(wrong) > 0:
        alpha[wrong, i] -= signs[wrong]
    
    for j, classifier in enumerate(wrong):
        if classifier == True: 
            non_zero_alphas[j] = np.append(non_zero_alphas[j], i)

    return(alpha, non_zero_alphas)

'''
# non_zero_alphas = [np.flatnonzero(alpha[classifier]) for classifier in range(alpha.shape[0])]
'''