'''

@author wronk

This plots a histogram of the clustering coefficient probability and clustering
coefficient by brain area
'''

import matplotlib.pyplot as plt
import numpy as np


def plot_clustering_coeff_pdf(coeffs, bins=np.linspace(0, 1, 50)):
    '''
    Plot clustering coefficient probability density function
    '''

    # Normalize coefficients

    # Constuct figure
    #fig, (ax0, ax1) = plt.subplots(ncols=1)
    fig = plt.figure()

    # Plot coefficients according to bins
    plt.hist(coeffs.flatten(), bins, histtype='step', fc='g', alpha=1,
             normed=True)
    plt.title('Clustering Coefficient PDF')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Probability')

    return fig


def plot_clustering_coeff_ranked(coeffs, names, num_ranked=10):
    '''
    Plot clustering coefficient ranked by maximum value
    '''

    # Constuct figure
    fig = plt.figure()

    # Plot top ranked coefficients according to bins
    fig.bar(coeffs, names, fc='green', alpha=1)
    fig.set_title('Ranked Clustering Coefficients')
    fig.set_xlabel('Region')
    fig.set_ylabel('Clustering Coefficient')

    return fig


if __name__ == '__main__':

    import scipy.io as sio
    import os.path as op

    weights_dir = '/home/wronk/Builds/friday-harbor/linear_model/'
    D_W_ipsi = sio.loadmat(op.join(weights_dir, 'W_ipsi.mat'))
    D_W_contra = sio.loadmat(op.join(weights_dir, 'W_ipsi.mat'))
    D_PValue_ipsi = sio.loadmat(op.join(weights_dir, 'W_ipsi.mat'))
    D_PValue_contra = sio.loadmat(op.join(weights_dir, 'W_ipsi.mat'))

    plt.ion()

    fig = plot_clustering_coeff_pdf(D_W_ipsi['data'])
    plt.show()
