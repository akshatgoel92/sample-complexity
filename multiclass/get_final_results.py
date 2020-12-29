import os
import pickle
import helpers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def compute_final_cf(n_classes = 10, question_no='1.2', id=31, dest = '../results/confusion_matrix.csv'):
    '''
    Post-process the final CF into the format required by
    the question
    '''
    results = helpers.open_results(question_no, id)
    
    test_cf = results['test_cf']
    
    cf = np.zeros((n_classes, n_classes))

    for result in test_cf:
        np.fill_diagonal(result, 0)
        cf = np.add(cf, result)

    cf = pd.DataFrame(np.divide(cf, np.sum(cf))*100)

    cf.to_csv(dest)

    return(cf)


def process_frequent_mistakes(question_no, id):
    '''
    Store images of frequent mistakes
    '''
    results = helpers.open_results(question_no, id)

    mistakes = results['mistakes']
    mistakes = np.vstack(np.unique(np.concatenate(mistakes), return_counts = True)).T

    sorted_mistakes = mistakes[np.argsort(mistakes[: ,1])]
    sorted_mistakes = sorted_mistakes[-5:, 0]

    X, Y = helpers.load_data('data', 'zipcombo.dat')

    for i, img_no in sorted_mistakes:
        show_images(X[img_no], path = 'results/{}_{}_{}.png'.format(img_no, Y[img_no], sorted_mistakes[i, 1]))