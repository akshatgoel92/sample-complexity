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


def run_test_case(perceptron_args, n_classes):
    '''
    --------------------------------------
    Execute the training steps above and generate
    the results that have been specified in the report.
    --------------------------------------
    '''
    X_train, Y_train = helpers.load_data("data", "dtrain123.dat")
    X_val, Y_val = helpers.load_data("data", "dtest123.dat")

    Y_train = Y_train - 1
    Y_val = Y_val - 1

    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    history = train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, n_classes)

    return(history)


if __name__ == '__main__':

  n_classes = 2

  perceptron_args = {
    
        'epochs': 20, 
        'kernel_type': 'polynomial', 
        'n_classifiers': 1,
        'tolerance': 0.000001,
        'convergence_epochs': 5,
        'tolerance': 0.0000001, 
        'd':3, 
        'fit_type': 'one_vs_one',
        'question_no': '3_4'
    }


run_test_case(perceptron_args, n_classes)

  


 
