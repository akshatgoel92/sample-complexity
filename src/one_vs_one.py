import kernel_perceptron as perceptron
import helpers
import numpy as np


def train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, n_classes=3):
  '''
  Input:
  Output:
  '''
  n_samples = len(Y_train)
  predictions = np.array([np.empty(n_samples)])
  votes = np.zeros((n_classes, len(Y_val)))
  
  for i in range(n_classes):
    for j in range(i+1, n_classes):
      
      X_train_subset = X_train[(Y_train == i) | (Y_train == j)]
      Y_train_subset = Y_train[(Y_train == i) | (Y_train == j)]
      
      history = perceptron.train_perceptron(X_train, Y_train, X_val, Y_val, **perceptron_args, fit_type='one_vs_one')
      prediction = history['preds_val']

      i_votes = np.zeros(len(Y_val))
      i_votes[prediction > 0] = 1
      votes[i] += i_votes
      
      j_votes = np.zeros(len(Y_val))
      j_votes[prediction <= 0] = 1
      votes[j] += j_votes
  
  return np.argmax(votes, axis=0)


if __name__ == '__main__':


  perceptron_args = {
    
        'epochs': 20, 
        'kernel_type': 'polynomial', 
        'n_classes': 2,
        'tolerance': 0.000001,
        'convergence_epochs': 5,
        'sparse_setting': 0,
        'tolerance': 0.0000001, 
        'd':3
    }

  X_train, Y_train = helpers.load_data("data", "dtrain123.dat")
  X_val, Y_val = helpers.load_data("data", "dtest123.dat")
  
  Y_train = Y_train.astype(int)
  Y_val = Y_val.astype(int)

  train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args)


'''
For testing
'''
'''
epochs = 20
kernel_type = 'polynomial'
n_classifiers = 1
tolerance = 0.000001
convergence_epochs = 5
sparse_setting = 0
tolerance = 0.000001
d = 3
'''
