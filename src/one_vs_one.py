import kernel_perceptron as perceptron
import helpers
import numpy as np


def train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, n_classes):
  '''
  Input:
  Output:
  '''
  n_samples = len(Y_train)
  predictions = np.array([np.empty(n_samples)])
  votes = np.zeros((n_classes, len(Y_val)))
  count = 0
  for i in range(n_classes):
    for j in range(i+1, n_classes):
      history = perceptron.train_perceptron(X_train, Y_train, X_val, Y_val, **perceptron_args, neg=i, pos=j)
      count += 1
      print(count)

  return history


if __name__ == '__main__':

  n_classes = 2

  perceptron_args = {
    
        'epochs': 20, 
        'kernel_type': 'polynomial', 
        'n_classifiers': 1,
        'tolerance': 0.000001,
        'convergence_epochs': 5,
        'sparse_setting': 0,
        'tolerance': 0.0000001, 
        'd':3, 
        'fit_type': 'one_vs_one',
    }

  X_train, Y_train = helpers.load_data("data", "dtrain123.dat")
  X_val, Y_val = helpers.load_data("data", "dtest123.dat")
  
  Y_train = Y_train.astype(int)
  Y_val = Y_val.astype(int)

  train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, n_classes)
