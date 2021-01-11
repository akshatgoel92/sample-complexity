import os
import pickle
from multiclass import helpers
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def open_results(question_no, id):
    '''
    Open results according to question no.
    '''
    f_name = os.path.join('results', '{}_results_id_{}.txt'.format(question_no, id))
    
    with open(f_name, 'rb') as f:
        results = pickle.load(f)

    return(results)


def compute_final_cf(n_classes = 10, question_no='1.2', id=31, 
                     dest = './results/confusion_matrix_correct.csv',
                     dest_std = "./results/confusion_matrix_std.csv"):
    '''
    Post-process the final CF into the format required by
    the question
    '''
    results = open_results(question_no, id)
    
    test_cf = results['test_cf']
    
    cf = np.zeros((n_classes, n_classes))

    for result in test_cf:
        cf = np.add(cf, result)

    cf = cf/cf.sum(axis=1, keepdims=1)*100
    np.fill_diagonal(cf, 0)
    
    for result in test_cf:
        result = result/result.sum(axis=1, keepdims=1)*100

    cf_std = np.std(np.array(test_cf), axis=0)
    np.fill_diagonal(cf_std, 0)

    pd.DataFrame(cf).to_csv(dest)
    pd.DataFrame(cf_std).to_csv(dest_std)

    return(cf, cf_std)


def process_frequent_mistakes(question_no = '1.1_polynomial_get_images_cv_mistakes', id=9, train=True):
    '''
    Store images of frequent mistakes
    '''
    results = open_results(question_no, id)

    if train: 
        results_type = 'train_mistakes'
        largest_n = 5
    else:
        results_type = 'mistakes'
        largest_n = 2

    results_name = results_type + '_worst_5'
    
    mistakes = np.vstack(np.unique(np.concatenate(results[results_type]), return_counts = True)).T
    
    sorted_mistakes = mistakes[np.argsort(mistakes[: ,1])]
    print(sorted_mistakes)
    
    sorted_mistakes = sorted_mistakes[-largest_n:, 0]
    print(sorted_mistakes)

    X, Y = helpers.load_data('data', 'zipcombo.dat')
    imgs = X[sorted_mistakes]
    print(Y[sorted_mistakes])
    
    plot_imgs(imgs, path = 'results/{}.png'.format(results_name))



def plot_imgs(imgs, shape=(16, 16), path='results/test.png'):
  '''
  Display imgs
  '''
  plt.figure(figsize=(20,10))
  columns = 5
  
  for i, img in enumerate(imgs):
    plt.subplot(len(imgs) / columns + 1, columns, i + 1)
    plt.imshow(img.reshape(shape))
  
  plt.savefig(path)


if __name__ == '__main__':

    # compute_final_cf()
    process_frequent_mistakes()
    process_frequent_mistakes(train=False)