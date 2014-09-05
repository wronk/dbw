import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: wronk

Plot saved lesion statistics
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle
from os import path as op

# Set relative directory path to linear model & ontology
stat_save_path = './cache'
fig_save_path = './cache'

stats_to_graph = ['avg_shortest_path', 'avg_eccentricity', 'avg_ccoeff']
lesion_attr = ['degree_labels', 'random']
network_names = ['allen', 'biophysical', 'scale_free']
num_lesions = 5

col = ['b', 'magenta', 'r']

# Construct matrix out of stats
stat_mat = np.zeros((len(stats_to_graph), len(lesion_attr), len(network_names),
                     num_lesions))

fig, ax = plt.subplots(nrows=1, ncols=1)
for l_atr_i, l_atr in enumerate(lesion_attr):
    for ni, net in enumerate(network_names):
        f_name = net + '_lesionBy_' + l_atr + 'x' + str(num_lesions) + \
            '_stats.pkl'
        #{'stats': graph_stats, 'stat_names': stats_to_graph,
        #'graph': network_type, # 'targeted': targeted_attack,
        #'target': lesion_attr, 'num_lesions': num_lesions})
        graph_stats = pickle.load(open(op.join(stat_save_path, f_name), 'rb'))

        for si, stat in enumerate(stats_to_graph):
            for li in range(num_lesions):
                stat_mat[si, l_atr_i, ni, li] = graph_stats['stats'][li][stat]

for si, stats in enumerate(stats_to_graph):
    fig, axes = plt.subplots(nrows=1, ncols=len(lesion_attr), squeeze=False)

    #if type(axes) is not np.ndarray and type(axes) is not list:
    #    axes = list(axes)

    for ai in range(axes.shape[1]):
        for ni in range(len(network_names)):
            axes[0, ai].scatter(range(num_lesions), stat_mat[si, ai, ni, :],
                                color=col[l_atr_i])
            axes[0, ai].set_ylabel(lesion_attr[ai])
            axes[0, ai].set_xlabel('# Nodes Lesioned')

            axes[0, ai].set_xlim([-0.5, num_lesions + 1])
    plt.savefig(op.join(fig_save_path, stats_to_graph[si] + '.png'))
