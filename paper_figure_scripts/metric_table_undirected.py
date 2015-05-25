"""
Created Sat May 23 14:11:50 2015

@author: wronk

Create a table of useful metrics for undirected standard graphs and model
"""

import os.path as op
import numpy as np
import networkx as nx

import extract.brain_graph
import random_graph.binary_undirected as bio_und
from random_graph import binary_undirected

###############################
# Parameters
###############################

# SET YOUR SAVE DIRECTORY
save_dir = '/home/wronk/Documents/dbw_figs/'

repeats = 2
graph_names = ['Mouse Connectome', 'Random', 'Small-World', 'Scale-Free']
metrics = [nx.average_clustering, nx.average_shortest_path_length]
brain_size = [7., 7., 7.]

# Initialize matrix to store metric values
met_arr = -1 * np.ones((len(graph_names), repeats, len(metrics)))


def calc_metrics(G, metrics):
    """Helper function to calculate list of (function) metrics"""
    metric_vals = np.zeros((len(metrics)))
    for fi, func in enumerate(metrics):
        metric_vals[fi] = func(G)

    return metric_vals

###############################
# Create graph/ compute metrics
###############################
#TODO: consider using a dict instead of checking for each graph type

# Load mouse connectivity graph
G_brain, _, _ = extract.brain_graph.binary_undirected()
n_nodes = G_brain.number_of_nodes()
n_edges = G_brain.number_of_edges()

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Calculate metrics specially because no repeats can be done on brain
brain_metrics = calc_metrics(G_brain, metrics)
for met_i, bm in enumerate(metrics):
    met_arr[graph_names.index('Mouse Connectome'), :, met_i] = bm(G_brain)

for rep in np.arange(repeats):
    # Amplified pref attachment model using gamma 0f 1.67
    if 'Amplified Pref. Attachment' in graph_names:
        G_Bio, _, _ = bio_und(n_nodes, n_edges, np.info, 1.67, brain_size)
        met_arr[graph_names.index('Amplified Pref. Attachment'), rep, :] = \
            calc_metrics(G_Bio, metrics)

    # Random Configuration model (random with fixed degree sequence)
    if 'Random' in graph_names:
        G_CM, _, _ = binary_undirected.random_simple_deg_seq(sequence=brain_degree,
                                                             brain_size=brain_size,
                                                             tries=100)
        met_arr[graph_names.index('Random'), rep, :] = \
            calc_metrics(G_CM, metrics)

    # Small-World (Watts-Strogatz) model with standard reconnection prob
    if 'Small-World' in graph_names:
        G_SW = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)),
                                       0.159)
        met_arr[graph_names.index('Small-World'), rep, :] = \
            calc_metrics(G_SW, metrics)

    # Scale-Free (Barabasi-Albert) graph
    if 'Scale-Free' in graph_names:
        G_SF = nx.barabasi_albert_graph(n_nodes,
                                        int(round(brain_degree_mean / 2.)))
        met_arr[graph_names.index('Scale-Free'), rep, :] = \
            calc_metrics(G_SF, metrics)

##########################
# Save metrics
##########################

# Save original array and version averaged across repeats
fname = op.join(save_dir, 'undirected_metrics')
np.save(fname + '_orig.npy', met_arr)
np.savetxt(fname + '_averaged.csv', met_arr.mean(1), format='%.3',
           delimiter=',')
