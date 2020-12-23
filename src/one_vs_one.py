import kernel_perceptron as perceptron
import helpers
import numpy as np
from itertools import combinations
from operator import itemgetter


def train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, all_combinations):
  '''
  Train one vs one classifer
  '''
  results = {}
  n_samples = len(Y_train)
  
  for i in range(n_classes):
    for j in range(i+1, n_classes):

      history = perceptron.train_perceptron(X_train, Y_train, X_val, Y_val, **perceptron_args, neg=i, pos=j)
      results[(i,j)] =  history

  return results


def predict(target, results, all_combinations):
  '''
  Prediction function for one vs. one classifier
  '''
  combos = list(filter(lambda x: target in x, all_combinations))
  results = list(itemgetter(*combos)(results))
  
  preds_train = np.argmax(np.vstack([result['final_Y_hat_train'] for result in results]), axis = 0)
  preds_val = np.argmax(np.vstack([result['final_Y_hat_val'] for result in results]), axis = 0)

  return(preds_train, preds_val)


def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking OvR decision function from OvO.
    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.
    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.
    confidences : array-like of shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.
    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``.
    """
    
    # Store the number of samples in the predictions
    n_samples = predictions.shape[0]

    # Create the vote matrix
    votes = np.zeros((n_samples, n_classes))

    # Create a matrix which holds the confidence level
    sum_of_confidences = np.zeros((n_samples, n_classes))

    # k indexes the colum we are at which is in this case which classifier's 
    # predictions we are considering
    # We start with the 0th classifier
    k = 0

    # For example for 4 classes
    # (i, j, k) = ((0, 1), 0)
    # (i, j, k) = ((0, 2), 1)
    # (i, j, k) = ((0, 3), 2)
    # (i, j, k) = ((1, 2), 3)
    # (i, j, k) = ((1, 3), 4)
    # (i, j, k) = ((2, 3), 5)
    
    # Go through each negative label
    # Note that i is the negative label
    for i in range(n_classes):
        # Go through each corresponding positive label
        for j in range(i + 1, n_classes):
            # The ith column contains the cumulative confidence for label i 
            # It gets minus the confidence added to it from the kth classifier
            # So if this confidence is a very large negative number than
            # we are confident that the negative label is the correct label
            # So - 1 * -1 turns into a large positive number
            sum_of_confidences[:, i] -= confidences[:, k]
            # This is the positive label
            # So if the confidence is a large positive number we are confident
            # that the j label is actually the correct label
            # We just add it to the jth column to add a large positive number
            # to it
            sum_of_confidences[:, j] += confidences[:, k]
            
            # Wherever the kth classifier predicts 0 we add one vote to
            # the corresponding training example 
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1

            # Move to the next classifier
            k += 1


def run_test_case(perceptron_args, n_classes, all_combinations):
    '''
    Execute the training steps above and generate
    the results that have been specified in the report
    '''
    X_train, Y_train = helpers.load_data("data", "dtrain123.dat")
    X_val, Y_val = helpers.load_data("data", "dtest123.dat")

    Y_train = Y_train - 1
    Y_val = Y_val - 1

    Y_train = Y_train.astype(int)
    Y_val = Y_val.astype(int)

    history = train_one_vs_one(X_train, Y_train, X_val, Y_val, perceptron_args, all_combinations)

    return(history)





if __name__ == '__main__':

  n_classes = 3
  all_combinations  = combinations(range(n_classes), 2)

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

  results = run_test_case(perceptron_args, n_classes, all_combinations)