"""
Created Sat May 24 19:13:24 2015

@author: wronk

Create a table of useful metrics for directed standard graphs and model
"""

import os.path as op
import numpy as np
import networkx as nx

import extract.brain_graph
import random_graph.binary_directed as bio_dir

###############################
# Parameters
###############################

# SET YOUR SAVE DIRECTORY
save_dir = '/home/wronk/Documents/dbw_figs/'

repeats = 2
graph_names = ['Mouse Connectome', 'Pref. Growth, Prox. Attachment', 'Random']
metrics = [nx.assortativity.degree_assortativity_coefficient,
           nx.average_shortest_path_length]
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
G_brain, _, _ = extract.brain_graph.binary_directed()
n_nodes = G_brain.number_of_nodes()
n_edges = G_brain.number_of_edges()

# Calculate degree & clustering coefficient distribution
brain_indegree = G_brain.in_degree().values()
brain_outdegree = G_brain.out_degree().values()

# Calculate metrics specially because no repeats can be done on brain
brain_metrics = calc_metrics(G_brain, metrics)
for met_i, bm in enumerate(metrics):
    met_arr[graph_names.index('Mouse Connectome'), :, met_i] = bm(G_brain)

for rep in np.arange(repeats):
    # Amplified pref attachment model using gamma 0f 1.67
    if 'Pref. Growth, Prox. Attachment' in graph_names:
        G_Bio, _, _ = bio_dir.biophysical_reverse_outdegree(n_nodes, n_edges,
                                                            L=np.inf,
                                                            brain_size=brain_size)
        met_arr[graph_names.index('Amplified Pref. Attachment'), rep, :] = \
            calc_metrics(G_Bio, metrics)

    # Random Configuration model (random with fixed degree sequence)
    if 'Random' in graph_names:
        G_CM = bio_dir.random_directed_deg_seq(brain_indegree, brain_outdegree,
                                               simplify=True, tries=100)
        met_arr[graph_names.index('Random'), rep, :] = calc_metrics(G_CM,
                                                                    metrics)

    # Scale-Free (Barabasi-Albert) graph
    if 'Scale-Free' in graph_names:
        pass

##########################
# Save metrics
##########################

# Save original array and version averaged across repeats
fname = op.join(save_dir, 'directed_metrics')
np.save(fname + '_orig.npy', met_arr)
np.savetxt(fname + '_averaged.csv', met_arr.mean(1), format='%.3',
           delimiter=',')
