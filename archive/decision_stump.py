# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers as helpers


# We need to find a single pixel which satisfies the following conditions
# There are 10 threshold values such that the dataset when split according to 
# these values minimizes total Gini index at each split
# We can do this as follows: 
# 1) Each feature is a pixel from 1 to 256
# 2) For each feature: we cycle through each possible feature value from 1 to 255
# 3) For each feature value: we set that as threshold
# 4) We separate features into 2 groups based on this threshold
# 5) 


def test(train, i, val):

    train[]






def test_split(X, i, val):
    '''
    Take in an attribute and threshold value. 
    Return split dataset based on that threshold
    value.
    '''
    # Mask to check which point is below a given threshold
    below_val = (X[:, i] < val)
    # Return statement
    return(X[below_val], X[~below_val])


def get_gini_index(x):
    '''
    Calculate the Gini index given a single node's output.
    '''
    # No contribution to the coefficient for 0 sized groups
    if len(x) == 0: return 0.0
    # Else count the number of values in each bin
    counts = np.bincount(x)
    # Create the group-wise score
    p = counts / float(len(x))
    # Return the group-wise score
    return(1.0 - np.sum(p*p))


def get_overall_gini(gini_indices):
    '''
    Return size weighted average of all group-wise Gini indices.
    '''
    pass


def test_split(X, n_classes):
    '''
    This function iteratively
    tests each possible split of
    a given dataset and returns 
    the newly created groups based
    on the best split.
    '''
    class_values = range(n_classes)
    best_index, best_value, best_score, best_groups = 999, 999, 999, 999, None
    
    for i in range(len(X[0]) -1):
        for row in X:
            groups = get_split(X, i, row[index])
            gini = get_gini_index(groups)
            if gini < b_score:
                best_index, best_value, best_score, best_groups = i, row[i], gini, groups

    return({'index': best_index, 'value': best_value, 'groups': best_groups})



def to_terminal(group):
    '''
    Create a terminal node value
    '''
    # Pick the predicted value for each row in the group
    outcomes = [row[-1] for row in group]
    # Return statement
    return max(set(outcomes), key=outcomes.count)



def split(node, max_depth, min_size, depth):
    '''
    Create child splits for a node or make terminal
    '''
    
    # Get left and right splits
    left, right = node['groups']
    # Don't need this any more
    del(node['groups'])
    
    # Check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # Check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # Process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # Process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)



def build_tree(train, max_depth, min_size):
    '''
    Create a tree using recursive splitting
    '''
    # Get an initial split
    root = get_split(train)
    # Do binary recursive splitting
    split(root, max_depth, min_size, 1)
    # Return statement
    return root



if __name__ == '__main__':

    # This is where the dataset lives
    path = os.path.join('..', 'data')
    # This is the dataset name
    name = 'zipcombo.dat'
    # Load the data
    X, Y = helpers.load_data(path, name)
    # Test the Gini index function
    print(get_gini_index([1,0, 0, 0]))