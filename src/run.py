from kernel_perceptron import *



def run_test_case(epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                  check_convergence, kernel_type, d):
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

    settings = train_setup(X_train, Y_train, 
                           X_val, Y_val, fit_type, 
                           n_classifiers, d, kernel_type)

    history = train_perceptron(*settings, X_train, Y_train, X_val, Y_val, epochs, 
                               n_classifiers, question_no, convergence_epochs, fit_type, 
                               check_convergence)

    return(history)



def run_multiple(params, data_args, epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                 check_convergence, kernel_type, total_runs):
    '''
    --------------------------------------
    Run multiple runs of kernel 
    perceptron training with a given 
    set of parameters
    --------------------------------------
    '''
    results = []
    overall_run_no = 0

    time_msg = "Elapsed time is....{} minutes"
    start = time.time()

    for param in params:

        histories = {
        
        'params': param, 
        'history': [],
        'best_epoch': [],
        'best_training_loss': [],
        'best_dev_loss': [],
        }
        
        for run in range(total_runs):
            # Prepare data for the perceptron
            # Shuffle the dataset before splitting it
            # Split the data into training and validation set 
            X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
            X, Y = helpers.shuffle_data(X, Y)
            
            X_train, X_val, Y_train, Y_val = helpers.split_data(X, Y, data_args['train_percent'])
            
            # Convert data to integer
            Y_train = Y_train.astype(int)
            Y_val = Y_val.astype(int)


            settings = train_setup(X_train, Y_train,  X_val, Y_val, fit_type, n_classifiers, param, kernel_type)
            history = train_perceptron(*settings, X_train, Y_train, X_val, Y_val, epochs, 
                                        n_classifiers, question_no, convergence_epochs, fit_type, 
                                        check_convergence)

            # Call the perceptron training with the given epochs
            # Return best epoch according to dev. loss and the associated accuracies on both datasets
            best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(history)
            
            # Store results
            histories['best_training_loss'].append(best_training_loss)
            histories['best_dev_loss'].append(best_dev_loss)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(history)

            overall_run_no += 1
            print("This is overall run no {}".format(overall_run_no))
            elapsed = (time.time() - start)/60
            print(time_msg.format(elapsed))
        
        # Store results
        results.append(histories)
    
    helpers.save_experiment_results(results, question_no)
    
    return(histories)




def run_multiple_cv(params, data_args, epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                    check_convergence, kernel_type, total_runs):
    '''
    --------------------------------------
    Check which kernel parameter results
    in highest validation loss
    --------------------------------------
    '''
    results = []
    overall_run_no = 0

    time_msg = "Elapsed time is....{} minutes"
    start = time.time()
    
    for run in range(total_runs):


        histories = {
        
            'params': params, 
            'history': [],
            'best_epoch': [],
            'best_training_loss': [],
            'best_dev_loss': [],
            }

        # Prepare data for the perceptron
        X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
        X, Y = helpers.shuffle_data(X, Y)

        # Split into training and validation set
        X_train, X_test, Y_train, Y_test = helpers.split_data(X, Y, data_args['train_percent'])
        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)
        
        # Divide into a list of folds
        X_folds, Y_folds = helpers.get_k_folds(X_train, Y_train, data_args['k'])

        # Each fold will go here
        folds = []

        for fold_no in range(data_args['k']):
        
                # Put in the x-values
                X_train_fold = np.concatenate(X_folds[:fold_no] + X_folds[fold_no+1:])
                X_val_fold = X_folds[fold_no]
        
                # Put in the Y values
                Y_train_fold = np.concatenate(Y_folds[:fold_no] + Y_folds[fold_no+1:])
                Y_val_fold =  Y_folds[fold_no]

                # Append
                folds.append([X_train_fold, Y_train_fold, X_val_fold, Y_val_fold])

        # Now iterate through the parameters
        for param in params:

            # Print progress
            print("This is run {} for parameter d = {}...".format(run, param))

            # Get invariants per fold
            settings_list = [train_setup(*fold, fit_type, n_classifiers, param, kernel_type) 
                             for fold in folds]

            # Now go through each fold : every fold becomes the hold-out set at least once
            fold_histories = [train_perceptron(*settings, *fold, epochs, 
                                               n_classifiers, question_no, 
                                               convergence_epochs, fit_type, 
                                               check_convergence) for settings, fold in zip(settings_list, folds)]
        
            # Get avg. accuracies by epoch across folds
            avg_history = helpers.get_cv_results(fold_histories)
            best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(avg_history)
            
            # Append history
            histories['best_training_loss'].append(best_training_loss)
            histories['best_dev_loss'].append(best_dev_loss)
            histories['best_epoch'].append(best_epoch)
            histories['history'].append(avg_history)

        # Get best parameter value
        best_dev_config = np.argmin(np.array(histories['best_dev_loss']))
        best_param = histories['params'][best_dev_config]

        # Retrain
        print("Retraining now...")
        print("The best parameter is {}....".format(best_param))

        # We are ready to retrain
        history = train_perceptron(X_train, Y_train, 
                                   X_test, Y_test, 
                                   **cv_args, d=best_param, question_no=question_no)
        
        # Get retraining results
        best_epoch, best_training_loss, best_dev_loss = helpers.get_best_results(history)
        preds_train = history['preds_train'][best_epoch]
        preds_test = history['preds_val'][best_epoch]
        
        # Update the results
        histories['best_training_loss'] = [best_training_loss]
        histories['best_dev_loss'] = [best_dev_loss]
        histories['best_epoch'] = [best_epoch]
        histories['params'] = best_param
        histories['history'] = [history]
        histories['train_cf'] = helpers.get_confusion_matrix(Y_train, preds_train)
        histories['val_cf'] = helpers.get_confusion_matrix(Y_test, preds_test)

        overall_run_no += 1
        print("This is overall run no {}".format(overall_run_no))
        elapsed = (time.time() - start)/60
        print(time_msg.format(elapsed))

        # Append the results
        results.append(histories)

    # Save the results
    helpers.save_results(results, question_no)
    helpers.save_experiment_results(results, question_no)
    
    return(results)



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='List the content of a folder')
    
    parser.add_argument('question_no',
                         type=str, 
                         help='Specify the question number...')
    
    args = parser.parse_args()
    
    question_no = args.question_no


    # Set random seed
    np.random.seed(13290138)

    
    # Store kernel parameter list to iterate over
    params = [1, 2, 3, 4, 5, 6, 7]

    
    # Store the arguments relating to the data set
    data_args = {

        'data_path': 'data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5,

        }


    if question_no == 'test':

        # Store test arguments
        test_args = {

            'epochs':20,
            'n_classifiers': 3,
            'question_no': 'test.txt',
            'convergence_epochs':5,
            'fit_type': 'one_vs_all',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'd':3,

            }


        history = run_test_case(**test_args)

    if question_no == '1.1':


        # Store arguments for this
        multiple_run_args = {
    
            'epochs': 20, 
            'n_classifiers': 10,
            'question_no': '1.1',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'total_runs': 20 
        }

        run_multiple(params, data_args, **multiple_run_args)

    if question_no == '1.2':

        cv_args = {
    
            'epochs': 18,
            'n_classifiers': 10, 
            'question_no': '1.2',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': False,
            'kernel_type': 'polynomial',
            'total_runs': 20 
        }

        run_multiple_cv(params, data_args, **cv_args)


    if question_no == '1.4':


        # Store arguments for this
        multiple_run_args = {
    
            'epochs': 20, 
            'n_classifiers': 10,
            'question_no': '1.1',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': True,
            'kernel_type': 'gaussian',
            'total_runs': 20 
        }

        run_multiple(params, data_args, **multiple_run_args)

        
        cv_args = {
    
            'epochs': 18,
            'n_classifiers': 10, 
            'question_no': '1.2',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': False,
            'kernel_type': 'gaussian',
            'total_runs': 20 
        }

        run_multiple_cv(params, data_args, **cv_args)


    if question_no == '1.4.1':


        # Store arguments for this
        multiple_run_args = {
    
            'epochs': 20, 
            'n_classifiers': 10,
            'question_no': '1.1',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_one',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'total_runs': 20 
        }

        run_multiple(params, data_args, **multiple_run_args)


    if question_no == '1.4.2':
        
        cv_args = {
    
            'epochs': 20,
            'n_classifiers': 10, 
            'question_no': '1.2',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_one',
            'check_convergence': False,
            'kernel_type': 'polynomial',
            'total_runs': 20 
        }

        run_multiple_cv(params, data_args, **cv_args)