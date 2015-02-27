"""
Created on Thu Feb 26 12:29:20 2015

@author: wronk

Create figures showing progressive percolation on standard and brain graphs.
"""
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from extract import brain_graph
#import network_gen as ng
from metrics import percolation as perc
reload(perc)
from random_graph import binary_undirected as bu
import in_out_plot_config as cf
import os
import os.path as op
import pickle

prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
#node_order = 426

##############################################################################
### Load graphs
load_dir = '../cache'
graph_names_und = ['Mouse', 'Random', 'Small-World', 'Scale-Free',
                   'Biophysical']
graph_names_dir = ['Mouse', 'Random', 'Biophysical']

graph_metrics_und = []
graph_metrics_dir = []

### Metrics arrays are (function/metric x number/prop removed x repeats)

# Load undirected graph metrics
for g_name in graph_names_und:
    load_fname = op.join(load_dir, g_name + '_undirected.pkl')
    open_file = open(load_fname, 'rb')
    graph_metrics_und.append(pickle.load(open_file))

# Load directed graph metrics
for g_name in graph_names_dir:
    load_fname = op.join(load_dir, g_name + '_directed.pkl')
    open_file = open(load_fname, 'rb')
    graph_metrics_dir.append(pickle.load(open_file))

# Calculate mean and std dev across repeats
for g_dict in graph_metrics_und:
    g_dict['data_rand_avg'] = np.nanmean(g_dict['data_rand'], axis=-1)
    g_dict['data_rand_std'] = np.nanstd(g_dict['data_rand'], axis=-1)
    g_dict['data_targ_avg'] = np.nanmean(g_dict['data_target'], axis=-1)
    g_dict['data_targ_std'] = np.nanstd(g_dict['data_target'], axis=-1)
    g_dict['removed_rand'] = prop_rm  # Temporary hack
    g_dict['removed_targ'] = lesion_list  # Temporary hack
for g_dict in graph_metrics_dir:
    g_dict['data_rand_avg'] = np.nanmean(g_dict['data_rand'], axis=-1)
    g_dict['data_rand_std'] = np.nanstd(g_dict['data_rand'], axis=-1)
    g_dict['data_targ_avg'] = np.nanmean(g_dict['data_target'], axis=-1)
    g_dict['data_targ_std'] = np.nanstd(g_dict['data_target'], axis=-1)
    g_dict['removed_rand'] = prop_rm  # Temporary hack
    g_dict['removed_targ'] = lesion_list  # Temporary hack
##############################################################################
### Plot results
# Set font type for compatability with adobe if doing editing later
plt.close('all')
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

graph_col = ['k', 'r', 'g', 'b', 'c']
LW = 3.
FONTSIZE = cf.FONTSIZE
FIGSIZE = (11, 5.5)

#######################################
### Random attack (undirected) with subplot for each metric
fig1, ax_list1 = plt.subplots(nrows=1,
                              ncols=len(graph_metrics_und[0]['metrics_list']),
                              figsize=FIGSIZE)

for fi, func in enumerate(graph_metrics_und[0]['metrics_list']):
    for gi, g_dict in enumerate(graph_metrics_und):
        x = g_dict['removed_rand']
        avg = g_dict['data_targ_avg'][fi, :]  # TODO Reverse after fix
        fill_upper = avg + g_dict['data_targ_std'][fi, :]
        fill_lower = avg - g_dict['data_targ_std'][fi, :]

        ax_list1[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        ax_list1[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list1[fi].set_title('Random Attack', fontsize=FONTSIZE)
    ax_list1[fi].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
    ax_list1[fi].set_ylabel(['Largest Component Size',
                             'Average Shortest Path'][fi],
                            fontsize=FONTSIZE)

ax_list1[1].legend(loc=0)

for ax in ax_list1:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Random attack (directed) with subplot for each metric

#######################################
### Targeted attack (undirected) with subplot for each metric

#######################################
### Targeted attack (directed) with subplot for each metric
'''
#######################################
### Combined plot hack
fig4, ax_list4 = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

for gi in np.arange(len(graph_names)):
    ax_list4[0].plot(lesion_list, metrics_target_avg[0, gi, :] * node_order,
                     lw=LW, label=graph_names[gi], color=graph_col[gi])

for gi in np.arange(len(graph_names)):
    ax_list4[1].plot(lesion_list, metrics_target_avg[1, gi, :],
                     label=graph_names[gi], color=graph_col[gi], lw=LW)

ax_list4[0].set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax_list4[0].set_xlabel('Number Nodes Removed', fontsize=FONTSIZE)
ax_list4[0].set_ylabel('Size of Largest\nRemaining Cluster', fontsize=FONTSIZE)
ax_list4[0].set_xlim((0, 400))
ax_list4[0].set_ylim((0, 450))
ax_list4[0].locator_params(axis='x', nbins=5)
ax_list4[0].locator_params(axis='y', nbins=5)
ax_list4[0].text(.95, .95, 'a', ha='right', va='top', fontsize=FONTSIZE,
                 transform=ax_list4[0].transAxes, weight='bold')

ax_list4[1].set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax_list4[1].set_xlabel('Number Nodes Removed', fontsize=FONTSIZE)
ax_list4[1].set_ylabel('Avg. Shortest\nGeodesic Distance', fontsize=FONTSIZE)
ax_list4[1].legend(loc=2, fontsize=FONTSIZE - 4.5, labelspacing=0.25,
                   borderpad=0.25)
ax_list4[1].set_xlim((0, 350))
ax_list4[1].set_ylim((1, 8))
ax_list4[1].locator_params(axis='x', nbins=5)
ax_list4[1].locator_params(axis='y', nbins=8)
ax_list4[1].text(.925, .95, 'b', ha='right', va='top', fontsize=FONTSIZE,
                 transform=ax_list4[1].transAxes, weight='bold')

for ax in ax_list4:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)


'''
#######################################
fig1.tight_layout()
#fig2.tight_layout()

plt.show()
