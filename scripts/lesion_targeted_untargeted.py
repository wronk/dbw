"""
Created on Mon Nov 24 12:29:20 2014

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

repeats = 100  # Number of times to repeat percolation
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
node_order = 426

linear_model_path = '/home/wronk/Builds/friday-harbor/linear_model'

##############################################################################


def construct_graph_list(graphs_to_const):
    ''' Construct and return a list of graphs so graph construction is easily
    repeatable'''

    graph_check = ['Erdos-Renyi', 'Small-World', 'Scale-Free', 'Biophysical']
    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_AL, _, _ = brain_graph.binary_undirected()
    graph_list.append(G_AL)

    # Calculate degree & clustering coefficient distribution
    n_nodes = G_AL.order()
    n_edges = float(len(G_AL.edges()))
    p_edge = n_edges / ((n_nodes * (n_nodes - 1)) / 2.)

    brain_degree = nx.degree(G_AL).values()
    brain_degree_mean = np.mean(brain_degree)

    # Construct Random (ER) graph
    if graph_check[0] in graphs_to_const:
        graph_list.append(nx.erdos_renyi_graph(n_nodes, p_edge))

    # Construct WS graph
    if graph_check[1] in graphs_to_const:
        graph_list.append(nx.watts_strogatz_graph(
            n_nodes, int(round(brain_degree_mean)), 0.159))

    # Construct BA graph
    if graph_check[2] in graphs_to_const:
        graph_list.append(nx.barabasi_albert_graph(
            n_nodes, int(round(brain_degree_mean / 2.))))

    # Construct biophysical graph
    if graph_check[3] in graphs_to_const:
        G_BIO, _, _ = bu.biophysical(N=n_nodes, N_edges=n_edges, L=2.2,
                                     gamma=1.67)
        graph_list.append(G_BIO)

    assert len(graph_list) == len(graph_names), 'Graph list/names don\'t match'

    return graph_list

##############################################################################
### Construct graphs
graph_names = ['Mouse Connectome', 'Erdos-Renyi', 'Small-World', 'Scale-Free',
               'Biophysical Model']
#graph_names = ['Mouse', 'Erdos-Renyi']
graph_col = ['k', 'r', 'g', 'b', 'c']
graph_list = construct_graph_list(graph_names)

func_list = [perc.lesion_met_largest_component,
             perc.lesion_met_avg_shortest_path,
             perc.lesion_met_diameter]

##############################################################################
### Do percolation
metrics_rand = np.zeros((repeats, len(func_list), len(graph_names),
                         len(prop_rm)))
metrics_target = np.zeros((repeats, len(func_list), len(graph_names),
                           len(lesion_list)))

S_target = np.zeros((repeats, len(graph_names), len(lesion_list)))
asp_target = np.zeros((repeats, len(graph_names), len(lesion_list)))

# Percolation
for ri in np.arange(repeats):
    print 'Graph cycle ' + str(ri + 1) + '/' + str(repeats)
    graph_list = construct_graph_list(graph_names)
    for gi, G in enumerate(graph_list):
        print '\tLesioning: ' + graph_names[gi]
        metrics_rand[ri, :, gi, :] = perc.percolate_random(G, prop_rm,
                                                           func_list)
        metrics_target[ri, :, gi, :] = perc.percolate_degree(G, lesion_list,
                                                             func_list)

metrics_rand_avg = np.nanmean(metrics_rand, axis=0)
metrics_target_avg = np.nanmean(metrics_target, axis=0)

##############################################################################
### Save results
# TODO
##############################################################################
### Plot results
# Set font type for compatability with adobe if doing editing later
plt.close('all')
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['pdf.fonttype'] = 42
plt.ion()

LW = 3.
FONTSIZE = 20
FIGSIZE = (11, 5.5)

#######################################
### Largest cluster
fig1, ax_list1 = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=FIGSIZE)

for gi in np.arange(len(graph_names)):
    ax_list1[0].plot(prop_rm, metrics_rand_avg[0, gi, :], lw=LW,
                     label=graph_names[gi], color=graph_col[gi])
for gi in np.arange(len(graph_names)):
    ax_list1[1].plot(lesion_list, metrics_target_avg[0, gi, :], lw=LW,
                     label=graph_names[gi], color=graph_col[gi])

# Set title and labels
ax_list1[0].set_title('Random Attack', fontsize=FONTSIZE)
ax_list1[0].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
ax_list1[0].set_ylabel('Normalized Cluster Size', fontsize=FONTSIZE)

ax_list1[1].set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax_list1[1].set_xlabel('Number Nodes Removed', fontsize=FONTSIZE)
ax_list1[1].set_ylabel('Normalized Cluster Size', fontsize=FONTSIZE)
ax_list1[1].legend(loc=0)

for ax in ax_list1:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Average shortest path length
fig2, ax_list2 = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

for gi in np.arange(len(graph_names)):
    ax_list2[0].plot(prop_rm, metrics_rand_avg[1, gi, :],
                     label=graph_names[gi], color=graph_col[gi], lw=LW)
for gi in np.arange(len(graph_names)):
    ax_list2[1].plot(lesion_list, metrics_target_avg[1, gi, :],
                     label=graph_names[gi], color=graph_col[gi], lw=LW)

# Set title and labels
ax_list2[0].set_title('Random Attack', fontsize=FONTSIZE)
ax_list2[0].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
ax_list2[0].set_ylabel('Avg. Shortest\nGeodesic Distance', fontsize=FONTSIZE)

ax_list2[1].set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax_list2[1].set_xlabel('Number Nodes Removed', fontsize=FONTSIZE)
ax_list2[1].set_ylabel('Avg. Shortest\nGeodesic Distance', fontsize=FONTSIZE)
ax_list2[1].legend(loc=2)

for ax in ax_list2:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

#######################################
### Average diameter
fig3, ax_list3 = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)

for gi in np.arange(len(graph_names)):
    ax_list3[0].plot(prop_rm, metrics_rand_avg[2, gi, :], lw=LW,
                     label=graph_names[gi], color=graph_col[gi])
for gi in np.arange(len(graph_names)):
    ax_list3[1].plot(lesion_list, metrics_target_avg[2, gi, :], lw=LW,
                     label=graph_names[gi], color=graph_col[gi])

# Set title and labels
ax_list3[0].set_title('Random Attack', fontsize=FONTSIZE)
ax_list3[0].set_xlabel('Proportion Nodes Removed', fontsize=FONTSIZE)
ax_list3[0].set_ylabel('Avg. Diameter', fontsize=FONTSIZE)

ax_list3[1].set_title('Targeted Attack (by Degree)', fontsize=FONTSIZE)
ax_list3[1].set_xlabel('Number Nodes Removed', fontsize=FONTSIZE)
ax_list3[1].set_ylabel('Avg. Diameter', fontsize=FONTSIZE)
ax_list3[1].legend(loc=0)

for ax in ax_list3:
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)

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


fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

plt.show()
