from multiclass import helpers
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sammon import sammon 
import os

from sklearn.manifold import MDS



if __name__ == '__main__':

    np.random.seed(13289)

    X, Y = helpers.load_data('data', 'zipcombo.dat')

    X_shuffle, Y_shuffle, perm = helpers.shuffle_data(X, Y)

    X_train, _, target, _, _, _ = helpers.split_data(X_shuffle, Y_shuffle, perm, 0.3)

    viz_type = 'sammon'

    n_classes = 10

    
    if viz_type == 'MDS':

        model = MDS(n_components=2, dissimilarity='euclidean', random_state=1, max_iter=100)
        y = model.fit_transform(X_train)

    if viz_type =='sammon':

        n = 2
        [y, E] = sammon(X_train, n)



    x = np.arange(n_classes)
    ys = [i+x+(i*x)**2 for i in range(n_classes)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for i in range(n_classes):
        plt.scatter(y[target==i, 0], y[target==i, 1], s=20, color=colors[i], alpha = 0.6, marker='o',label=names[i])


    plt.title('{} projection of scanned digits dataset'.format(viz_type.title()))
    plt.legend(loc='best')
    plt.savefig(os.path.join('figs', viz_type + '.png'))
    plt.show()
