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
from network_plot.change_settings import (set_all_text_fontsizes,
                                          set_all_colors)

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
    open_file.close()

# Load directed graph metrics
for g_name in graph_names_dir:
    load_fname = op.join(load_dir, g_name + '_directed.pkl')
    open_file = open(load_fname, 'rb')
    graph_metrics_dir.append(pickle.load(open_file))
    open_file.close()

# Calculate mean and std dev across repeats
for g_dict in graph_metrics_und:
    g_dict['data_rand_avg'] = np.mean(g_dict['data_rand'], axis=-1)
    g_dict['data_rand_std'] = np.std(g_dict['data_rand'], axis=-1)
    g_dict['data_targ_avg'] = np.mean(g_dict['data_targ'], axis=-1)
    g_dict['data_targ_std'] = np.std(g_dict['data_targ'], axis=-1)
for g_dict in graph_metrics_dir:
    g_dict['data_rand_avg'] = np.mean(g_dict['data_rand'], axis=-1)
    g_dict['data_rand_std'] = np.std(g_dict['data_rand'], axis=-1)
    g_dict['data_targ_avg'] = np.mean(g_dict['data_targ'], axis=-1)
    g_dict['data_targ_std'] = np.std(g_dict['data_targ'], axis=-1)
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
FACECOLOR = cf.FACECOLOR
LABELCOLOR = cf.LABELCOLOR
TICKSIZE = cf.TICKSIZE

#######################################
### Random attack (undirected) with subplot for each metric
# construct figure
fig1, ax_list1 = plt.subplots(nrows=1,
                              ncols=len(graph_metrics_und[0]['metrics_list']),
                              facecolor=FACECOLOR,
                              figsize=FIGSIZE)

# Loop over each metric and then each graph
for fi, func_label in enumerate(graph_metrics_und[0]['metrics_label']):
    for gi, g_dict in enumerate(graph_metrics_und):
        # Compute x axis vals, y vals, and std devs
        x = g_dict['removed_rand']
        avg = g_dict['data_rand_avg'][fi, :]
        fill_upper = avg + g_dict['data_rand_std'][fi, :]
        fill_lower = avg - g_dict['data_rand_std'][fi, :]

        # Plot traces
        ax_list1[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        # Plot std deviation range
        ax_list1[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list1[fi].set_title('Random Attack', fontsize=FONTSIZE)
    ax_list1[fi].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
    ax_list1[fi].set_ylabel(func_label, fontsize=FONTSIZE)

ax_list1[1].legend(loc=0)

for ax in ax_list1:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Random attack (directed) with subplot for each metric
fig2, ax_list2 = plt.subplots(nrows=1,
                              ncols=len(graph_metrics_dir[0]['metrics_list']),
                              facecolor=FACECOLOR,
                              figsize=FIGSIZE)

for fi, func_label in enumerate(graph_metrics_dir[0]['metrics_label']):
    for gi, g_dict in enumerate(graph_metrics_dir):
        x = g_dict['removed_rand']
        avg = g_dict['data_rand_avg'][fi, :]
        fill_upper = avg + g_dict['data_rand_std'][fi, :]
        fill_lower = avg - g_dict['data_rand_std'][fi, :]

        ax_list2[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        ax_list2[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list2[fi].set_title('Random Attack', fontsize=FONTSIZE)
    ax_list2[fi].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
    ax_list2[fi].set_ylabel(func_label, fontsize=FONTSIZE)

ax_list2[1].legend(loc=0)

for ax in ax_list2:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Targeted attack (undirected) with subplot for each metric
fig3, ax_list3 = plt.subplots(nrows=1,
                              ncols=len(graph_metrics_und[0]['metrics_list']),
                              facecolor=FACECOLOR,
                              figsize=FIGSIZE)

for fi, func_label in enumerate(graph_metrics_und[0]['metrics_label']):
    for gi, g_dict in enumerate(graph_metrics_und):
        x = g_dict['removed_targ']
        avg = g_dict['data_targ_avg'][fi, :]
        fill_upper = avg + g_dict['data_targ_std'][fi, :]
        fill_lower = avg - g_dict['data_targ_std'][fi, :]

        ax_list3[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        ax_list3[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list3[fi].set_title('Targeted Attack', fontsize=FONTSIZE)
    ax_list3[fi].set_xlabel('Number of Nodes Removed', fontsize=FONTSIZE)
    ax_list3[fi].set_ylabel(func_label, fontsize=FONTSIZE)

ax_list3[1].legend(loc=0)

for ax in ax_list3:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Targeted attack (directed) with subplot for each metric
fig4, ax_list4 = plt.subplots(nrows=1,
                              ncols=len(graph_metrics_dir[0]['metrics_list']),
                              facecolor=FACECOLOR,
                              figsize=FIGSIZE)

for fi, func_label in enumerate(graph_metrics_dir[0]['metrics_label']):
    for gi, g_dict in enumerate(graph_metrics_dir):
        x = g_dict['removed_targ']
        avg = g_dict['data_targ_avg'][fi, :]
        fill_upper = avg + g_dict['data_targ_std'][fi, :]
        fill_lower = avg - g_dict['data_targ_std'][fi, :]

        ax_list4[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=graph_col[gi])
        ax_list4[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=graph_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list4[fi].set_title('Targeted Attack', fontsize=FONTSIZE)
    ax_list4[fi].set_xlabel('Number of Nodes Removed', fontsize=FONTSIZE)
    ax_list4[fi].set_ylabel(func_label, fontsize=FONTSIZE)

ax_list4[1].legend(loc=0)

for ax in ax_list4:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Combined plot hack
fig_col = ['m', 'r', 'orange']
fig5, ax_list5 = plt.subplots(nrows=1, ncols=2, figsize=(9, 3.75),
                              facecolor=FACECOLOR, tight_layout=True)

for fi, func_label in enumerate(graph_metrics_dir[0]['metrics_label'][0:2]):
    for gi, g_dict in enumerate(graph_metrics_dir):
        if g_dict['graph_name'] == 'Biophysical':
            g_dict['graph_name'] = 'Model'
        x = g_dict['removed_targ']
        avg = g_dict['data_targ_avg'][fi, :]
        fill_upper = avg + g_dict['data_targ_std'][fi, :]
        fill_lower = avg - g_dict['data_targ_std'][fi, :]

        ax_list5[fi].plot(x, avg, lw=LW, label=g_dict['graph_name'],
                          color=fig_col[gi])
        ax_list5[fi].fill_between(x, fill_upper, fill_lower, lw=0,
                                  facecolor=fig_col[gi], interpolate=True,
                                  alpha=.3)
    ax_list5[fi].set_title('Targeted Lesion')
    ax_list5[fi].set_xlabel('# Nodes Removed')
    ax_list5[fi].set_ylabel(['Largest Strong\nComponent',
                             'Avg. Shortest Path'][fi])

#ax_list5[0].set_xlim((0, 400))
#ax_list5[0].set_ylim((0, 450))

leg = ax_list5[0].legend(loc=1, fontsize=FONTSIZE - 4., labelspacing=0.2,
                         borderpad=0.15)
leg.get_frame().set_alpha(0.)
for text in leg.get_texts():
    text.set_color(LABELCOLOR)
#ax_list5[1].set_xlim((0, 350))
#ax_list5[1].set_ylim((1, 8))
ax_list5[0].xaxis.set_major_locator(plt.MaxNLocator(4))
ax_list5[0].yaxis.set_major_locator(plt.MaxNLocator(4))
ax_list5[1].xaxis.set_major_locator(plt.MaxNLocator(4))

for temp_ax in ax_list5:
    set_all_text_fontsizes(temp_ax, FONTSIZE + 2)
    set_all_colors(temp_ax, LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=TICKSIZE)

fig5.savefig('/home/wronk/Builds/lesion_fig_poster.png', transparent=True)

'''
#######################################
fig1.tight_layout()
#fig2.tight_layout()

plt.show()
'''
