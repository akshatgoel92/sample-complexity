import json
import numpy as np 
import matplotlib.pyplot as plt
from linear_regression import *



def run_sample_complexity(n, 
                          min_m, 
                          model_name = 'linear_regression', 
                          max_iter=100, 
                          step_size=1, 
                          n_test=20, 
                          target_test_error=0.1, 
                          size_test=100):
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

    X, Y = helpers.get_binary_data(min_m + max_iter, n)
    
    
    for m in range(min_m, max_iter, step_size):

        X_train = X[:m, :]
        Y_train = Y[:m]
        

        if model_name == 'linear_regression':
            model = LinearRegression(X_train,Y_train)
        
        train_error = model.fit()
        all_test_errors = [model.validate(X_test, Y_test) for X_test, Y_test in test_sets]
        avg_test_error = np.mean(np.array(all_test_errors))

        history["m"].append(m)
        history["avg_test_error"].append(avg_test_error)

        if avg_test_error <= target_test_error:
            
            converged = True
            history["converged"] = converged
            history["final_m"] = m
            
            return(history)
    
    history["converged"] = False
    print("Did not converge...please adjust the step size or increase the max iterations!")

    return(history)


def create_new_experiment_file(model_name = 'linear_regression'):
    '''
    Create new experiment file
    '''
    new_experiment = []
    new_experiment_path = os.path.join('results', '{}.json'.format(model_name))

    with open(new_experiment_path, 'w') as f:
        json.dump(new_experiment, f)




def run_experiment(model_name = "linear_regression", max_n=100):
    '''
    Load existing experiment file, run experiment, append results,
    write experiment file
    '''
    
    experiment = get_experiment(model_name)

    # Only converged results are in experiment
    start_n = len(experiment) + 1

    for n in range(start_n, max_n + 1):
        history = run_sample_complexity(n, min_m = 1)
        if history['converged']:
            experiment.append(history)

    write_experiment(model_name, experiment)




def write_experiment(model_name, experiment):
    '''
    Write experiment to file
    '''

    experiment_path = os.path.join('results', '{}.json'.format(model_name))

    with open(experiment_path, 'w') as f:
        json.dump(experiment, f)


def get_experiment(model_name = "linear_regression"):
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


def plot_experiment(model_name = "linear_regression"):
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

    np.random.seed(182390)
    new = 1
    
    #if new == 1:
        # create_new_experiment_file()
    
    # run_experiment()
    plot_experiment()