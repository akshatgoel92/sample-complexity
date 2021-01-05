import json
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *
from vanilla_perceptron import *
from one_nn import *
from winnow import *



def run_sample_complexity(n, 
                          min_m, 
                          model_name, 
                          max_iter=100, 
                          step_size=1, 
                          n_test=20, 
                          n_train = 500,
                          target_test_error=0.1, 
                          size_test=100,
                          online_epochs=100):
    '''
    Run sample complexity code
    '''
    test_sets = [helpers.get_binary_data(size_test, n) for i in range(n_test)]

    avg_test_error = np.inf
    converged = False
    m = min_m

    # Store results here
    history = { "n": n, "max_iter": max_iter, 
                "step_size": step_size, 
                "n_test": n_test,
                "size_test": size_test,
                "m": [] , 
                "train_error": [], 
                "avg_test_error": [], 
                "converged": "", 
                "final_m":""}
    
    train_sets = [helpers.get_binary_data(min_m + max_iter, n) for i in range(n_train)]
    
    for m in range(min_m, max_iter, step_size):

        print('m' + str(m))

        current_train_m = [(X_train[:m, :], Y_train[:m]) for X_train, Y_train in train_sets]
        
        if model_name == 'linear_regression':
            models = [LinearRegression(X_train,Y_train) for X_train, Y_train in current_train_m]
        
        if model_name == 'linear_perceptron':
            models = [Perceptron(X_train,Y_train, online_epochs) for X_train, Y_train in current_train_m]

        if model_name == 'one_nn':
            models = [KNN(1, X_train, Y_train) for X_train, Y_train in current_train_m]

        if model_name == 'winnow':
            models = [Winnow(X_train, Y_train, online_epochs, n) for X_train, Y_train in current_train_m]


        if model_name != 'one_nn':
            train_errors = [model.fit() for model in models]
        
        all_test_errors = [[model.validate(X_test, Y_test) for X_test, Y_test in test_sets] for model in models]
        avg_test_errors = [np.mean(np.array(test_errors)) for test_errors in all_test_errors]

        history["m"].append(m)
        history["avg_test_error"].append(avg_test_errors)
        avg_test_error = np.mean(np.array(avg_test_errors))
        print(avg_test_error)

        if avg_test_error <= target_test_error:
            
            converged = True
            history["converged"] = converged
            history["final_m"] = m
            
            return(history)
    
    history["converged"] = False
    print("Did not converge...please adjust the step size or increase the max iterations!")

    return(history)


def create_new_experiment_file(model_name):
    '''
    Create new experiment file
    '''
    new_experiment = []
    new_experiment_path = os.path.join('results', '{}.json'.format(model_name))

    with open(new_experiment_path, 'w') as f:
        json.dump(new_experiment, f)




def run_experiment(model_name, max_n, online_epochs, min_m, max_iter):
    '''
    Load existing experiment file, run experiment, append results,
    write experiment file
    '''
    
    experiment = get_experiment(model_name)

    # Only converged results are in experiment
    start_n = len(experiment) + 1

    for n in range(start_n, max_n + 1):
        print(n)
        history = run_sample_complexity(n, 
                                        min_m, 
                                        model_name=model_name, 
                                        max_iter = max_iter,
                                        online_epochs=online_epochs)
        if history['converged']:
            experiment.append(history)

        if n%10 == 0: write_experiment(model_name, experiment)

    write_experiment(model_name, experiment)




def write_experiment(model_name, experiment):
    '''
    Write experiment to file
    '''

    experiment_path = os.path.join('results', '{}.json'.format(model_name))

    with open(experiment_path, 'w') as f:
        json.dump(experiment, f)


def get_experiment(model_name):
    '''
    Load a particular experiment
    '''
    experiment_path = os.path.join('results', '{}.json'.format(model_name))

    with open(experiment_path) as f:
        try:
            experiment = json.load(f)
        except ValueError:
            experiment = []

    return(experiment)


def plot_experiment(model_name):
    '''
    Make a plot of the experiment
    '''
    plot_path = os.path.join('results', '{}.png'.format(model_name))
    experiment = get_experiment(model_name)

    n = list(range(len(experiment)))
    m = [experiment_run['final_m'] for experiment_run in experiment]

    plt.plot(n, m)
    plt.xlabel("Dimension: n")
    plt.ylabel("Estimated Sample complexity: m")
    plt.savefig(plot_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='List the content of a folder')
    
    parser.add_argument('model_name',
                         type=str, 
                         help='Specify the question number...')

    parser.add_argument('new',
                         type=int, 
                         help='Specify the question number...')
    
    args = parser.parse_args()
    model_name = args.model_name
    new = args.new

    np.random.seed(182390)
    min_m = 1
    max_n = 100
    max_iter = 1000
    online_epochs=100

    if new == 1:
        create_new_experiment_file(model_name=model_name)
    
    run_experiment(model_name=model_name, max_n=max_n, online_epochs=online_epochs, min_m=min_m, max_iter=max_iter)
    plot_experiment(model_name=model_name)