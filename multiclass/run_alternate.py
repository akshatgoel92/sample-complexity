from knn import *
from samme import *  
import argparse
import time


def run_test_case():

    # Load full dataset
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])
    
    # Shuffle and split dataset
    X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X,Y)

    k = 3
    
    # Split dataset
    X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, data_args['train_percent'])

    # Call the perceptron training with the given epochs
    knn = KNN(k, X_train, Y_train, X_val, Y_val)

    history = knn.fit()

    print(history)

    return(history)


def run_multiple(params, data_args, total_runs, model, question_no, lr = 0.01, epochs=1, n_classes=10):
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

    # Load data
    X, Y = helpers.load_data(data_args['data_path'], data_args['name'])

    # Start timer
    overall_run_no = 0
    time_msg = "Elapsed time is....{} minutes"
    start = time.time()

    # Store datasets here
    datasets = []
    
    # Create # total_runs different splits of the datasets 
    for run in range(total_runs):
        
        # Prepare data for the perceptron
        # Shuffle the dataset before splitting it
        # Split the data into training and validation set 
        X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X, Y)
        X_train, X_val, Y_train, Y_val, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, 
                                                                  data_args['train_percent'])
            
        # Convert data to integer
        Y_train = Y_train.astype(int)
        Y_val = Y_val.astype(int)

        # Store data splits and settings in tuples
        data = (X_train, Y_train, X_val, Y_val)

        # Apend dataset for this run to the list of run-wise datasets
        datasets.append(data)
        
    # Start run
    for param in params:

        histories = {
        
        'param': param, 
        'train_loss': [],
        'val_loss': [],

        }

        for run, data in enumerate(datasets):
            # Prepare data for the KNN
            # Shuffle the dataset before splitting it
            # Split the data into training and validation set 
            # Now train
            if model == 'knn':
            # Get settings for training
                knn = KNN(param, *data)
                history = knn.fit()
                train_loss = history['train_loss'][0]
                val_loss = history['val_loss'][0]

            if model == 'samme':
                samme = SAMME(lr, epochs, n_classes, param, *data)
                history = samme.fit()
                train_loss = history['train_loss'][0]
                val_loss = history['val_loss'][0]
            
            # Store results and mistakes
            histories['train_loss'].append(train_loss)
            histories['val_loss'].append(val_loss)
            
            overall_run_no += 1
            print("This is overall run no {} for parameter k = {}".format(overall_run_no, param))
            elapsed = (time.time() - start)/60
            print(time_msg.format(elapsed))

        # Append results
        results['train_loss_mean'].append(np.mean(np.array(histories['train_loss'])))
        results['train_loss_std'].append(np.std(np.array(histories['train_loss'])))
        results['val_loss_mean'].append(np.mean(np.array(histories['val_loss'])))
        results['val_loss_std'].append(np.std(np.array(histories['val_loss'])))

    # Save mistakes separately
    helpers.save_experiment_results(results, question_no)
    
    return(results)


def run_multiple_cv(params, data_args, total_runs, question_no, model, lr=0.01, epochs=1, n_classes=10):
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



            # Now go through each fold : every fold becomes the hold-out set at least once
            if model == 'knn':
                knn_estimators = [KNN(param, *fold) for fold in folds]
                fold_histories = [knn.fit() for knn in knn_estimators]

            if model == 'samme':
                samme_estimators = [SAMME(lr, epochs, n_classes, param, *fold) for fold in folds]
                fold_histories = [samme.fit() for samme in samme_estimators]
        
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

        # We are ready to retrain
        if model == 'knn':
            best_knn = KNN(best_param, X_train, Y_train, X_test, Y_test)
            history = best_knn.fit()

        if model == 'samme':
            best_samme = SAMME(lr, epochs, n_classes, best_param, X_train, Y_train, X_test, Y_test)
            history = best_samme.fit()
        
        # Get retraining results and append
        results['best_param'].append(best_param)
        results['train_loss'].append(history['train_loss'][0])
        results['test_loss'].append(history['val_loss'][0])

        overall_run_no += 1
        print("This is overall run no {}".format(overall_run_no))
        elapsed = (time.time() - start)/60
        print(time_msg.format(elapsed))
    
    # Save the table as a .csv
    helpers.save_experiment_results(results, question_no)
    
    return(results)


if __name__ == '__main__':

    np.random.seed(13290320)

    parser = argparse.ArgumentParser(description='List the content of a folder')
    
    parser.add_argument('question_no',
                         type=str, 
                         help='Specify the question number...')
    
    args = parser.parse_args()
    question_no = args.question_no

    data_args = {

        'data_path': './data',
        'name': 'zipcombo.dat', 
        'train_percent': 0.8,
        'k': 5}

    if question_no == 'test':

        run_test_case()

    if question_no == 'table_24':

        np.random.seed(2312319)

        params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        multiple_run_args = {
    
            'total_runs': 5,
            'model': 'knn',
            'question_no': question_no
        }

        run_multiple(params, data_args, **multiple_run_args)


    if question_no == 'table_17':

        params = [1, 2, 3, 4, 5, 6, 7]

        multiple_run_args = {
    
            'total_runs': 20,
            'model': 'knn',
            'question_no': question_no
        }

        run_multiple(params, data_args, **multiple_run_args)


    if question_no == 'table_19':

        params = [1, 2, 3, 4, 5, 6, 7]

        cv_args = {
            
            'question_no': question_no,
            'total_runs': 20,
            'model': 'knn' 
        }

        run_multiple_cv(params, data_args, **cv_args)


    if question_no == 'table_25':

        np.random.seed(372123)

        params = list(np.arange(2, 100, 2))

        multiple_run_args = {
    
            'total_runs': 5,
            'model': 'samme',
            'question_no': question_no
        }

        run_multiple(params, data_args, **multiple_run_args)

    if question_no == 'table_20':

        params = list(np.arange(60, 100, 2))

        multiple_run_args = {
    
            'total_runs': 20,
            'model': 'samme',
            'question_no': question_no
        }

        run_multiple(params, data_args, **multiple_run_args)

    if question_no == 'table_21':

        params = list(np.arange(60, 100, 2))
        cv_args = {
            
            'question_no': question_no,
            'total_runs': 20,
            'model': 'samme' 
        }

        run_multiple_cv(params, data_args, **cv_args)