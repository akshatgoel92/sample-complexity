import kernel_perceptron as perceptron
import numpy as np
import helpers
import time


def subset_data(X_train, Y_train, X_val, Y_val, n_classes):
  '''
  Get data subsets for 1 v 1 classifier
  '''
  data = []
  masks = []
  tracker = {}

  k = 0

  for i in range(n_classes):
    for j in range(i+1, n_classes):

      tracker[k] = (i,j)

      train_mask = (Y_train == i) | (Y_train == j)
      val_mask = (Y_val == i) | (Y_val == j)

      # Subset the data here
      X_train_sub = X_train[train_mask]
      Y_train_sub = Y_train[train_mask]
      
      X_val_sub = X_val[val_mask]
      Y_val_sub = Y_val[val_mask]

      datasets = (X_train_sub, Y_train_sub, X_val_sub, Y_val_sub)
      bool_masks = (train_mask, val_mask)
      
      data.append(datasets)
      masks.append(bool_masks)

      k+=1

  return(data, tracker, masks)


def train_one_vs_one(datasets, tracker, masks, n_train, n_val,
                     epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                     check_convergence, kernel_type, d, n_classes, Y_train, Y_val):
  '''
  Train one vs one classifer
  '''
  
  train_predictions = np.zeros((n_classifiers, n_train))
  train_confidences = np.zeros((n_classifiers, n_train))
  
  val_predictions = np.zeros((n_classifiers, n_val))
  val_confidences = np.zeros((n_classifiers, n_val))

  train_votes = np.zeros((n_classifiers, n_train))
  val_votes = np.zeros((n_classifiers, n_val))

  val_total_confidences = np.zeros((n_classifiers, n_val))
  train_total_confidences = np.zeros((n_classifiers, n_train))
  
  
  classifiers_per_training = 1
  
  for k, data in enumerate(datasets):

      
      i, j = tracker[k]
      train_mask, val_mask = masks[k]

      settings = perceptron.train_setup(*data, fit_type, 
                                        classifiers_per_training, d, kernel_type, neg = i, pos = j)

                  
      # Now train
      history = perceptron.train_perceptron(*settings, *data, epochs, 
                                             classifiers_per_training, 
                                             question_no, convergence_epochs, fit_type, 
                                             check_convergence, neg=i, pos=j)

      # Store predictions and confidences
      train_predictions[k, train_mask] = history['preds_train']
      train_confidences[k, train_mask] = history['Y_hat_train']
      
      # Store the validation predictions
      val_predictions[k, val_mask] = history['preds_val']
      val_confidences[k, val_mask] = history['Y_hat_val']


      # Update results
      results = update_results(train_total_confidences, val_total_confidences, 
                               train_votes, val_votes, train_confidences, 
                               val_confidences, train_predictions, 
                               val_predictions, i, j, k)

      # Unpack updated results
      train_total_confidences, val_total_confidences, train_votes, val_votes = results

  # Return predictions
  train_preds = np.argmax(train_total_confidences, axis = 0)
  val_preds = np.argmax(val_total_confidences, axis = 0)

  train_loss = helpers.get_loss(train_preds, Y_train)
  val_loss = helpers.get_loss(val_preds, Y_val)

  return train_loss, val_loss


def update_results(train_total_confidences, val_total_confidences, 
                   train_votes, val_votes, train_confidences, 
                   val_confidences, train_predictions, 
                   val_predictions, i, j, k):
    '''
    Update results from kth classifier
    '''
    # Update the overall confidences and vote counts
    train_total_confidences[i, :] -= train_confidences[k, :]
    train_total_confidences[j, :] += train_confidences[k, :]
      
    # Update the confidences
    val_total_confidences[i, :] -= val_confidences[k, :]
    val_total_confidences[j, :] += val_confidences[k, :]

    # Update the votes
    train_votes[i, train_predictions[k, :] == -1] += 1
    train_votes[j, train_predictions[k, :] == 1] += 1
      
    val_votes[i, val_predictions[k, :] == -1] += 1
    val_votes[j, val_predictions[k, :] == 1] += 1

    return(train_total_confidences, 
           val_total_confidences, 
           train_votes, val_votes)



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
                                            epochs, n_classifiers, question_no, 
                                            convergence_epochs, fit_type, 
                                            check_convergence, kernel_type, 
                                            d, n_classes)
    

    return(train_loss, val_loss)


def run_multiple(params, data_args, epochs, n_classifiers, 
                 question_no, convergence_epochs, fit_type, 
                 check_convergence, kernel_type, total_runs, n_classes):
    '''
    Run multiple runs of kernel 
    perceptron training with a given 
    set of parameters
    --------------------------------------
    '''
    results = {'param': params,
               'train_loss_mean': [],
               'train_loss_std': [],
               'val_loss_mean': [],
               'val_loss_std': []}

    
    # Load data
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])


    # Start timer
    overall_run_no = 0
    time_msg = "Elapsed time is....{} minutes"
    start = time.time()

    
    # Start run
    for param in params:

        histories = {
        
        'param': param, 
        'train_loss': [],
        'val_loss': []

        }
        
        for run in range(total_runs):

            # Prepare data for multiple runs
            
            # Shuffle the dataset before splitting it and then split into training and validation set
            X_shuffle, Y_shuffle = helpers.shuffle_data(X,Y)
            X_train, X_val, Y_train, Y_val = helpers.split_data(X_shuffle, Y_shuffle, data_args['train_percent'])
            
            # Convert data to integer
            Y_train = Y_train.astype(int)
            Y_val = Y_val.astype(int)

            datasets, tracker, masks = subset_data(X_train, Y_train, X_val, Y_val, n_classes)

            n_train = len(Y_train)
            n_val = len(Y_val)
            
            # Now train
            train_loss, val_loss = train_one_vs_one(datasets, tracker, masks, n_train, n_val,
                                                    epochs, n_classifiers, question_no, 
                                                    convergence_epochs, fit_type, 
                                                    check_convergence, kernel_type, 
                                                    param, n_classes, Y_train, Y_val)
            
            # Store results
            histories['train_loss'].append(train_loss)
            histories['val_loss'].append(val_loss)

            overall_run_no += 1
            print("This is overall run no {} for parameter d = {}".format(overall_run_no, param))
            elapsed = (time.time() - start)/60
            print(time_msg.format(elapsed))

        # Append results
        results['train_loss_mean'].append(np.mean(np.array(histories['train_loss'])))
        results['train_loss_std'].append(np.std(np.array(histories['train_loss'])))
        results['val_loss_mean'].append(np.mean(np.array(histories['val_loss'])))
        results['val_loss_std'].append(np.std(np.array(histories['val_loss'])))

        print(results)
    
    helpers.save_experiment_results(results, question_no)
    
    return(results)



if __name__ == '__main__':

  question_no = 'multiple'

  if question_no == 'test':

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

  if question_no == 'multiple':

      # Store kernel parameter list to iterate over
      params = [1]

      # Store the arguments relating to the data set
      data_args = {

          'data_path': 'data',
          'name': 'zipcombo.dat', 
          'train_percent': 0.8,
          'k': 5,

          }


      multiple_run_args = {
    
            'epochs': 20, 
            'n_classifiers': 45,
            'question_no': '1.4.3',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_one',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'total_runs': 1, 
            'n_classes': 10
        }

      run_multiple(params, data_args, **multiple_run_args)




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