import kernel_perceptron as perceptron
import numpy as np
import helpers
import time


def train_one_vs_one(datasets, tracker, masks, n_train, n_val,
                     epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                     check_convergence, kernel_type, d, n_classes, Y_train, Y_val, K_train, K_val):
  '''
  Train one vs one classifer
  '''
  alpha_weights = np.zeros((n_classifiers, n_train))
  classifiers_per_training = 1
  
  for k, data in enumerate(datasets):

      i, j = tracker[k]
      train_mask, _ = masks[k]

      settings = perceptron.train_setup(*data, fit_type, 
                                        classifiers_per_training, 
                                        d, kernel_type, neg = i, pos = j)

      # Now train
      history = perceptron.train_perceptron(*settings, *data, epochs, 
                                             classifiers_per_training, 
                                             question_no, convergence_epochs, fit_type, 
                                             check_convergence, neg=i, pos=j)

      # Store predictions and confidences
      alpha_weights[k, train_mask] = history['alpha']

  # Return predictions
  train_confidences = alpha_weights @ K_train
  val_confidences = alpha_weights @ K_val
  train_preds, val_preds = get_predictions(alpha_weights, K_train, K_val)
  
  train_loss = helpers.get_loss(train_preds, Y_train)
  val_loss = helpers.get_loss(val_preds, Y_val)

  print("Overall train loss {}, Overall val loss {}".format(train_loss, val_loss))

  return train_loss, val_loss


def get_predictions(alpha_weights, tracker, train_confidences, val_confidences):
    '''
    Update results from kth classifier
    '''
    k = 0
    
    for i in range(n_classes):
        for j in range(i+1, n_classes)

        # Check whether we have the right classifier at each iteration
        assert tracker[k] == (i, j)
        
        # Update the overall confidences and vote counts
        train_total_confidences[i, :] -= train_confidences[k, :]
        train_total_confidences[j, :] += train_confidences[k, :]
      
        # Update the confidences
        val_total_confidences[i, :] -= val_confidences[k, :]
        val_total_confidences[j, :] += val_confidences[k, :]
        k += 1

    train_preds = np.argmax(train_total_confidences, axis=0)
    val_preds = np.argmax(val_total_confidences, axis = 0)

    return(train_preds, val_preds)


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



def run_test_case_one_vs_one(epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
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

    n_train = len(Y_train)
    n_val = len(Y_val)

    datasets, tracker, masks = subset_data(X_train, Y_train, X_val, Y_val, n_classes)

    # Call the training function
    train_loss, val_loss = train_one_vs_one(datasets, tracker, masks, n_train, n_val,
                                            epochs, n_classifiers, question_no, 
                                            convergence_epochs, fit_type, 
                                            check_convergence, kernel_type, 
                                            d, n_classes, Y_train, Y_val)

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

    all_subset_datasets = []
    all_splits = []
    all_masks = []
    all_trackers = []
    all_gram = []

    for run in range(total_runs):

      # Prepare data for multiple runs
      # Shuffle the dataset before splitting it and then split into training and validation set
      X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X,Y)
      X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

            
      # Convert data to integer
      Y_train = Y_train.astype(int)
      Y_val = Y_val.astype(int)

      K_train = helpers.get_polynomial_kernel(X_train, X_train)
      K_val = helpers.get_polynomial_kernel(X_train, X_val)

      datasets, tracker, masks = subset_data(X_train, Y_train, X_val, Y_val, n_classes)
      all_splits.append((X_train, X_val, Y_train, Y_val))
      all_subset_datasets.append(datasets)
      all_trackers.append(tracker)
      all_masks.append(masks)
      all_gram.append((K_train, K_val))

    # These remains constant over different splits
    n_train = len(Y_train)
    n_val = len(Y_val)

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
        
        for run, datasets in enumerate(all_subset_datasets):

          tracker = all_trackers[run]
          masks = all_masks[run]
          _, _, Y_train, Y_val = all_splits[run]
            
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


def run_multiple_cv(params, data_args, epochs, n_classifiers, 
                    question_no, convergence_epochs, fit_type, 
                    check_convergence, kernel_type, total_runs):
    '''Check which kernel parameter results
    in lowest validation loss
    --------------------------------------
    '''
    
    results = {'best_param': [],
               'train_loss': [],
               'test_loss': [],
               }

    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
    overall_run_no = 0

    time_msg = "Elapsed time is....{} minutes"
    start = time.time()
    
    for run in range(total_runs):


        histories = {
                        'params': params,
                        'train_loss': [],
                        'val_loss': []
                    }

        # Store number of classes
        n_classes=10

        # Prepare data for the perceptron
        X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X, Y)

        # Split into training and validation set
        X_train, X_test, Y_train, Y_test, train_perm, test_perm = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])
        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)

        n_train = len(X_train)
        n_test = len(X_test)

        # For retraining best parameters
        full_datasets, full_tracker, full_masks = subset_data(X_train, Y_train, X_test, Y_test, n_classes)
        
        # Divide into a list of folds
        X_folds, Y_folds = helpers.get_k_folds(X_train, Y_train, data_args['k'])

        # Each fold will go here
        
        subset_datasets_by_fold = []
        splits_by_fold = []
        masks_by_fold = []
        trackers_by_fold = []

        for fold_no in range(data_args['k']):
        
                # Put in the x-values
                X_train_fold = np.concatenate(X_folds[:fold_no] + X_folds[fold_no+1:])
                X_val_fold = X_folds[fold_no]
        
                # Put in the Y values
                Y_train_fold = np.concatenate(Y_folds[:fold_no] + Y_folds[fold_no+1:])
                Y_val_fold =  Y_folds[fold_no]

                data, tracker, masks = subset_data(X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, n_classes)
                subset_datasets_by_fold.append(data)
                
                splits_by_fold.append([X_train_fold, X_val_fold, Y_train_fold, Y_val_fold])
                masks_by_fold.append(masks)
                trackers_by_fold.append(tracker)



        # Now iterate through the parameters
        for param in params:

            # Print progress
            print("This is run {} for parameter d = {}...".format(run, param))

            train_loss_by_fold = []
            val_loss_by_fold = []

            for fold, datasets in enumerate(subset_datasets_by_fold):

              tracker = trackers_by_fold[fold]
              masks = masks_by_fold[fold]
              _, _, Y_train_fold, Y_val_fold = splits_by_fold[fold]

              n_train_fold = len(Y_train_fold)
              n_val_fold = len(Y_val_fold)
            
              # Now train
              train_loss, val_loss = train_one_vs_one(datasets, tracker, masks, n_train_fold, n_val_fold,
                                                      epochs, n_classifiers, question_no, 
                                                      convergence_epochs, fit_type, 
                                                      check_convergence, kernel_type, 
                                                      param, n_classes, Y_train_fold, Y_val_fold)

              train_loss_by_fold.append(train_loss)
              val_loss_by_fold.append(val_loss)
        
            # Get avg. accuracies by epoch across folds
            cv_train_loss = np.mean(np.array(train_loss_by_fold))
            cv_val_loss = np.mean(np.array(val_loss_by_fold))
            
            # Append to the histories dictionary
            histories['train_loss'].append(cv_train_loss)
            histories['val_loss'].append(cv_val_loss)

        # Get best parameter value
        best_val_loss = np.argmin(np.array(histories['val_loss']))
        best_param = histories['params'][best_val_loss]

        # Retrain
        print("Retraining now...")
        print("The best parameter is {}....".format(best_param))

        
        # We are ready to retrain
        best_train_loss, best_test_loss = train_one_vs_one(full_datasets, full_tracker, full_masks, 
                                                           n_train, n_test,
                                                           epochs, n_classifiers, question_no, 
                                                           convergence_epochs, fit_type, 
                                                           check_convergence, kernel_type, 
                                                           param, n_classes, Y_train, Y_test)
        
        # Get retraining results and append
        results['best_param'].append(best_param)
        results['train_loss'].append(best_train_loss)
        results['test_loss'].append(best_test_loss)

        overall_run_no += 1
        print("This is overall run no {}".format(overall_run_no))
        elapsed = (time.time() - start)/60
        print(time_msg.format(elapsed))

    print(results)

    # Save the table as a .csv
    helpers.save_experiment_results(results, question_no)
    
    return(results)



if __name__ == '__main__':

  
  np.random.seed(1123)

  question_no = 'multiple_one_vs_one'

  if question_no == 'test':

    n_classes = 3
  
    train_args = {

              'epochs':5,
              'n_classifiers': 3,
              'question_no': 'test.txt',
              'convergence_epochs':5,
              'fit_type': 'one_vs_one',
              'check_convergence': True,
              'kernel_type': 'polynomial',
              'd':3,

        }

    train_loss, val_loss = run_test_case_one_vs_one(**train_args)
    print(train_loss, val_loss)

  

  if question_no == 'multiple_one_vs_one':

      # Store kernel parameter list to iterate over
      params = [1, 2, 3, 4, 5, 6, 7]

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
            'question_no': question_no,
            'convergence_epochs': 2,
            'fit_type': 'one_vs_one',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'total_runs': 20, 
            'n_classes': 10
        }

      results = run_multiple(params, data_args, **multiple_run_args)

  
  if question_no == 'cv_one_vs_one':

    # Store kernel parameter list to iterate over
    params = [1, 2, 3, 4, 5, 6, 7]

    # Store the arguments relating to the data set
    data_args = {

        'data_path': 'data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5,

        }

    params = [1, 2, 3, 4, 5, 6, 7]

    print(params)

    cv_args = {
    
        'epochs': 20,
        'n_classifiers': 45, 
        'question_no': question_no,
        'convergence_epochs': 2,
        'fit_type': 'one_vs_one',
        'check_convergence': True,
        'kernel_type': 'polynomial',
        'total_runs': 20 
    }

    results = run_multiple_cv(params, data_args, **cv_args)