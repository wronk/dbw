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
stats_to_graph_label = ['Avg Shortest Path', 'Avg Eccentricity',
                        'Avg Clustering Coeff']
lesion_attr = ['degree_labels', 'random']
lesion_attr_title = ['Target High Degree Nodes', 'Target Nodes Randomly']
network_names = ['allen', 'biophysical', 'scale_free']
num_lesions = 150

title_ft = 22
label_ft = title_ft - 4
tick_ft = 14

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
    fig, axes = plt.subplots(nrows=1, ncols=len(lesion_attr), figsize=(12, 6),
                             squeeze=False, sharey=True)

    #if type(axes) is not np.ndarray and type(axes) is not list:
    #    axes = list(axes)

    for ai in range(axes.shape[1]):
        for ni in range(len(network_names)):
            axes[0, ai].scatter(range(num_lesions), stat_mat[si, ai, ni, :],
                                color=col[ni], s=10)
            axes[0, ai].set_xlim([-0.25, num_lesions - 0.75])

            axes[0, ai].set_xlabel('# Nodes Lesioned', fontsize=label_ft)
            if ai == 0:
                axes[0, ai].set_ylabel(stats_to_graph_label[si],
                                       fontsize=label_ft)

            axes[0, ai].set_title(lesion_attr_title[ai],
                                  fontsize=title_ft, va='bottom')
            axes[0, ai].tick_params(labelsize=tick_ft)
            axes[0, ai].grid()

    plt.savefig(op.join(fig_save_path, stats_to_graph[si] + '.png'))
