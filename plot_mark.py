'''

@author wronk

This plots a histogram of the clustering coefficient probability and clustering
coefficient by brain area
'''

import matplotlib.pyplot as plt
import numpy as np


def plot_clustering_coeff_pdf(coeffs, bins=np.linspace(0., 0.25, 150)):
    '''
    Plot clustering coefficient probability density function
    '''

    # Normalize coefficients

    # Constuct figure
    #fig, (ax0, ax1) = plt.subplots(ncols=1)
    fig = plt.figure()

    # Plot coefficients according to bins
    plt.hist(coeffs.flatten(), bins, fc='g', alpha=.8, normed=True)
    plt.title('Clustering Coefficient PDF')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Probability')

    return fig


def plot_clustering_coeff_ranked(coeffs, names, num_ranked=10):
    '''
    Plot clustering coefficient ranked by maximum value
    '''
    # Graph params
    width = 0.5
    xpos = np.arange(num_ranked)

    # Constuct figure
    fig = plt.figure()

    sorted_tups = sorted(zip(coeffs, names), key=lambda tup: tup[0],
                         reverse=True)[:num_ranked]

    # Plot top ranked coefficients according to bins
    plt.bar(xpos, [w for w, _ in sorted_tups], fc='green',
            width=width, alpha=.8)
    plt.xticks(xpos + width / 2., [n for _, n in sorted_tups])

    plt.title('Ranked Clustering Coefficients')
    plt.xlabel('Region')
    plt.ylabel('Clustering Coefficient')

    return fig

if __name__ == '__main__':

    import scipy.io as sio
    import os.path as op

    weights_dir = '/home/wronk/Builds/friday-harbor/linear_model/'
    D_W_ipsi = sio.loadmat(op.join(weights_dir, 'W_ipsi.mat'))

    plt.ion()

    names = D_W_ipsi['row_labels']
    coeffs = np.random.rand(len(names))

    #fig = plot_clustering_coeff_pdf(D_W_ipsi['data'])
    #fig = plot_clustering_coeff_ranked(coeffs, names)

    # Graph params
    num_ranked=10
    width = 0.5
    xpos = np.arange(num_ranked)

    # Constuct figure
    fig = plt.figure()

    sorted_tups = sorted(zip(coeffs, names), key=lambda tup: tup[0],
                         reverse=True)[:num_ranked]

    # Plot top ranked coefficients according to bins
    plt.bar(xpos, [w for w, _ in sorted_tups], fc='green',
            width=width, alpha=.8)
    plt.xticks(xpos+width/2., [n for _, n in sorted_tups])

    plt.title('Ranked Clustering Coefficients')
    plt.xlabel('Region')
    plt.ylabel('Clustering Coefficient')
