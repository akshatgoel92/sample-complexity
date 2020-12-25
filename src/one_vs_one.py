import kernel_perceptron as perceptron
from itertools import combinations
from operator import itemgetter
import numpy as np
import helpers


def subset_data():
  '''
  Get data subsets for 1 v 1 classifier
  '''
  data = []
  tracker = {}

  for i in range(n_classes):
    for j in range(i+1, n_classes):

      tracker[k] = (i,j)

      
      # Subset the data here
      X_train_sub = X_train[(Y_train == i) | (Y_train == j)]
      Y_train_sub = Y_train[(Y_train == i) | (Y_train == j)]
      
      X_val_sub = X_val[(Y_val == i) | (Y_val == j)]
      Y_val_sub = Y_val[(Y_val == i) | (Y_val == j)]

      data.append(X_train_sub, Y_train_sub, X_val_sub, Y_val_sub)

  return(data, tracker)


def train_one_vs_one(X_train, Y_train, X_val, Y_val, 
                     epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                     check_convergence, kernel_type, d, n_classes):
  '''
  Train one vs one classifer
  '''
  train_predictions = np.zeros((n_classes, len(Y_train)))
  train_confidences = np.zeros((n_classes, len(Y_train)))
  val_predictions = np.zeros((n_classes, len(Y_val)))
  val_confidences = np.zeros((n_classes, len(Y_val)))

  # Store results here
  train_votes = np.zeros((n_classes, len(Y_train)))
  train_sum_of_confidences = np.zeros((n_classes, len(Y_train)))

  # 
  val_votes = np.zeros((n_classes, len(Y_val)))
  val_sum_of_confidences = np.zeros((n_classes, len(Y_val)))
  
  k = 0
  tracker = {}
  
  for i in range(n_classes):
    for j in range(i+1, n_classes):

      # Subset the data here
      tracker[k] = (i,j)

      X_train_sub = X_train[(Y_train == i) | (Y_train == j)]
      Y_train_sub = Y_train[(Y_train == i) | (Y_train == j)]
      
      X_val_sub = X_val[(Y_val == i) | (Y_val == j)]
      Y_val_sub = Y_val[(Y_val == i) | (Y_val == j)]

      settings = perceptron.train_setup(X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, fit_type, 
                                        n_classifiers, d, kernel_type, neg = i, pos = j)

                  
      # Now train
      history = perceptron.train_perceptron(*settings, X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, epochs, 
                                             n_classifiers, question_no, convergence_epochs, fit_type, 
                                             check_convergence, neg=i, pos=j)

      # Store predictions and confidences
      train_predictions[k, (Y_train == i) | (Y_train == j)] = history['preds_train']
      train_confidences[k, (Y_train == i) | (Y_train == j)] = history['Y_hat_train']
      
      # Store the validation predictions
      val_predictions[k, (Y_val == i) | (Y_val == j)] = history['preds_val']
      val_confidences[k, (Y_val == i) | (Y_val == j)] = history['Y_hat_val']

      # Update the overall confidences and vote counts
      train_sum_of_confidences[i, :] -= train_confidences[k, :]
      train_sum_of_confidences[j, :] += train_confidences[k, :]
      val_sum_of_confidences[i, :] -= val_confidences[k, :]
      val_sum_of_confidences[j, :] += val_confidences[k, :]

      train_votes[i, train_predictions[k, :] == -1] += 1
      train_votes[j, train_predictions[k, :] == 1] += 1
      val_votes[i, val_predictions[k, :] == -1] += 1
      val_votes[j, val_predictions[k, :] == 1] += 1

      k += 1

  # Return predictions using sum of
  train_preds = np.argmax(train_sum_of_confidences, axis = 0)
  val_preds = np.argmax(val_sum_of_confidences, axis = 0)

  train_loss = helpers.get_loss(train_preds, Y_train)
  val_loss = helpers.get_loss(val_preds, Y_val)

  return train_loss, val_loss



def predict(predictions, confidences, n_classes):
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
    n_samples = predictions.shape[1]

    # Create the vote matrix
    votes = np.zeros((n_classes, n_samples))

    # Create a matrix which holds the confidence level
    sum_of_confidences = np.zeros((n_classes, n_samples))

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
            sum_of_confidences[i, :] -= confidences[k, :]
            # This is the positive label
            # So if the confidence is a large positive number we are confident
            # that the j label is actually the correct label
            # We just add it to the jth column to add a large positive number
            # to it
            sum_of_confidences[j, :] += confidences[k, :]
            
            # Wherever the kth classifier predicts 0 we add one vote to
            # the corresponding training example 
            votes[i, predictions[k, :] == -1] += 1
            votes[j, predictions[k, :] == 1] += 1

            # Move to the next classifier
            k += 1

    return(np.argmax(votes, axis = 0), np.argmax(sum_of_confidences, axis = 0))



def run_test_case(epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                  check_convergence, kernel_type, d):
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

    # Call the training function
    history = train_one_vs_one(X_train, Y_train, X_val, Y_val, 
                               epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                               check_convergence, kernel_type, d, n_classes)
    
    # Call prediction function on training function
    train_votes_pred, train_conf_preds = predict(history[0], history[1], n_classes)
    val_votes_pred, val_conf_preds = predict(history[2], history[3], n_classes)

    # Store train loss and test loss
    train_loss = helpers.get_loss(train_conf_preds, Y_train)
    val_loss = helpers.get_loss(val_conf_preds, Y_val)

    return(train_loss, val_loss)



if __name__ == '__main__':

  n_classes = 3
  
  train_args = {

            'epochs':20,
            'n_classifiers': 1,
            'question_no': 'test.txt',
            'convergence_epochs':5,
            'fit_type': 'one_vs_one',
            'check_convergence': False,
            'kernel_type': 'polynomial',
            'd':3,

      }

  train_loss, val_loss = run_test_case(**train_args)
  print(train_loss, val_loss)




'''
import kernel_perceptron as perceptron
from itertools import combinations
from operator import itemgetter
import numpy as np
import helpers



n_classes = 3


X_train, Y_train = helpers.load_data("../data", "dtrain123.dat")
X_val, Y_val = helpers.load_data("../data", "dtest123.dat")

Y_train = Y_train - 1
Y_val = Y_val - 1

Y_train = Y_train.astype(int)
Y_val = Y_val.astype(int)

epochs = 20
n_classifiers = 1

question_no = 'test.txt'
convergence_epochs = 5

fit_type = 'one_vs_one'
check_convergence = False
kernel_type = 'polynomial'
d = 3

train_predictions = np.zeros((n_classifiers, len(Y_train)))
train_confidences = np.zeros((n_classifiers, len(Y_train)))
val_predictions = np.zeros((n_classifiers, len(Y_val)))
val_confidences = np.zeros((n_classifiers, len(Y_val)))
  
k = 0
tracker = {}

i = 0
j = 1

tracker[k] = (i,j)

X_train_sub = X_train[(Y_train == i) | (Y_train == j)]
Y_train_sub = Y_train[(Y_train == i) | (Y_train == j)]
      
X_val_sub = X_val[(Y_val == i) | (Y_val == j)]
Y_val_sub = Y_val[(Y_val == i) | (Y_val == j)]


settings = perceptron.train_setup(X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, fit_type, n_classifiers, d, kernel_type, neg = i, pos = j)

                  
# Now train
history = perceptron.train_perceptron(*settings, X_train_sub, Y_train_sub, X_val_sub, Y_val_sub, epochs, 
                                      n_classifiers, question_no, convergence_epochs, fit_type, 
                                      check_convergence, neg=i, pos=j)
'''