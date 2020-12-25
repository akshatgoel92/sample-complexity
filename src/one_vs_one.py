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
    train_loss, val_loss = train_one_vs_one(X_train, Y_train, X_val, Y_val, 
                                            epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                                            check_convergence, kernel_type, d, n_classes)
    

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