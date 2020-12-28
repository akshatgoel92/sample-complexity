# Imports
from kernel_perceptron import *




def run_test_case(epochs, n_classifiers, question_no, convergence_epochs, fit_type, 
                  check_convergence, kernel_type, d):
    '''
    --------------------------------------
    Execute the training steps above and generate
    the results that have been specified in the report.
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



def run_multiple(params, data_args, epochs, n_classifiers, 
                 question_no, convergence_epochs, fit_type, 
                 check_convergence, kernel_type, total_runs):
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
               'val_loss_std': []
               }


    all_mistakes = {

            'params': [], 
            'mistakes': []

            }

    
    # Load data
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])


    # Start timer
    overall_run_no = 0
    time_msg = "Elapsed time is....{} minutes"
    start = time.time()


    # Store datasets here
    datasets = []
    perms = []
    train_perms = []
    val_perms = []
    
    # Create # total_runs different splits of the datasets 
    for run in range(total_runs):
        
        # Prepare data for the perceptron
        # Shuffle the dataset before splitting it
        # Split the data into training and validation set 
        X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X, Y)
        X_train, X_val, Y_train, Y_val, train_perm, val_perm = helpers.split_data(X_shuffle, Y_shuffle, perm, 
                                                                                  data_args['train_percent'])
            
        # Convert data to integer
        Y_train = Y_train.astype(int)
        Y_val = Y_val.astype(int)

        # Store data splits and settings in tuples
        data = (X_train, Y_train, X_val, Y_val)

        # Apend dataset for this run to the list of run-wise datasets
        datasets.append(data)
        
        # Append permutations to the lists we created earlier for mistake tracking
        perms.append(perm)
        val_perms.append(val_perm)
        train_perms.append(train_perm)
        
    
    # Start run
    for param in params:

        histories = {
        
        'param': param, 
        'train_loss': [],
        'val_loss': [],

        }

        for run, data in enumerate(datasets):
            # Prepare data for the perceptron
            # Shuffle the dataset before splitting it
            # Split the data into training and validation set 
            # Now train

            # Get settings for training
            settings = train_setup(*data, fit_type, n_classifiers, param, kernel_type)

            history = train_perceptron(*settings, *data, epochs, 
                                        n_classifiers, question_no, 
                                        convergence_epochs, fit_type, 
                                        check_convergence)

            mistakes = helpers.get_mistakes(Y_val, history['preds_val'], val_perms[run])
            
            # Store results and mistakes
            histories['train_loss'].append(history['train_loss'])
            histories['val_loss'].append(history['val_loss'])

            all_mistakes['mistakes'].append(mistakes)
            all_mistakes['params'].append((param, run))
            
            overall_run_no += 1
            print("This is overall run no {} for parameter d = {}".format(overall_run_no, param))
            elapsed = (time.time() - start)/60
            print(time_msg.format(elapsed))


        # Append results
        results['train_loss_mean'].append(np.mean(np.array(histories['train_loss'])))
        results['train_loss_std'].append(np.std(np.array(histories['train_loss'])))
        results['val_loss_mean'].append(np.mean(np.array(histories['val_loss'])))
        results['val_loss_std'].append(np.std(np.array(histories['val_loss'])))

    # Save mistakes separately
    helpers.save_results(all_mistakes, question_no)
    helpers.save_experiment_results(results, question_no)
    
    return(results, all_mistakes)




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
               'train_cf': [],
               'test_cf': []
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

        # Prepare data for the perceptron
        X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X, Y)

        # Split into training and validation set
        X_train, X_test, Y_train, Y_test, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])
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
            cv_train_loss, cv_val_loss = helpers.get_cv_results(fold_histories)
            
            # Append to the histories dictionary
            histories['train_loss'].append(cv_train_loss)
            histories['val_loss'].append(cv_val_loss)

        # Get best parameter value
        best_val_loss = np.argmin(np.array(histories['val_loss']))
        best_param = histories['params'][best_val_loss]

        # Retrain
        print("Retraining now...")
        print("The best parameter is {}....".format(best_param))

        retrain_settings = train_setup(X_train, Y_train, X_test, Y_test, 
                                       fit_type, n_classifiers, best_param, kernel_type) 

        # We are ready to retrain
        history = train_perceptron(*retrain_settings, 
                                    X_train, Y_train, 
                                    X_test, Y_test, 
                                    epochs, n_classifiers, 
                                    question_no, convergence_epochs, 
                                    fit_type, check_convergence)
        
        # Get retraining results and append
        results['best_param'].append(best_param)
        results['train_loss'].append(history['train_loss'])
        results['test_loss'].append(history['val_loss'])
        results['train_cf'].append(helpers.get_confusion_matrix(Y_train, history['preds_train']))
        results['test_cf'].append(helpers.get_confusion_matrix(Y_test, history['preds_val']))

        overall_run_no += 1
        print("This is overall run no {}".format(overall_run_no))
        elapsed = (time.time() - start)/60
        print(time_msg.format(elapsed))

    # Save the results
    helpers.save_results(results, question_no)
    
    # Take the confusion matrix out before saving the table
    results.pop('train_cf', None)
    results.pop('test_cf', None)
    
    # Save the table as a .csv
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

            'epochs':3,
            'n_classifiers': 3,
            'question_no': 'test.txt',
            'convergence_epochs':5,
            'fit_type': 'one_vs_all',
            'check_convergence': False,
            'kernel_type': 'polynomial',
            'd':3,

            }


        history = run_test_case(**test_args)

    if '1.1' in question_no:


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

    if '1.2' in question_no:

        cv_args = {
    
            'epochs': 20,
            'n_classifiers': 10, 
            'question_no': '1.2',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': True,
            'kernel_type': 'polynomial',
            'total_runs': 20
        }

        run_multiple_cv(params, data_args, **cv_args)


    if '1.4' in question_no:


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


    if '1.4.1' in question_no:


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


    if '1.4.2' in question_no:
        
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


    if '1.5' in question_no:

        params = list(np.arange(0.01, 1.01, 0.01))

        print(params)

        multiple_run_args = {
    
            'epochs': 20, 
            'n_classifiers': 10,
            'question_no': '1.5',
            'convergence_epochs': 2,
            'fit_type': 'one_vs_all',
            'check_convergence': True,
            'kernel_type': 'gaussian',
            'total_runs': 1 
        }

        run_multiple(params, data_args, **multiple_run_args)