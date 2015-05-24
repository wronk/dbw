"""
Created Sat May 23 14:11:50 2015

@author: wronk

Create a table of useful metrics for undirected standard graphs and model
"""

import os.path as op
import numpy as np
import networkx as nx

import extract.brain_graph

###############################
# Parameters
###############################

# SET YOUR SAVE DIRECTORY
save_dir = '/home/wronk/Documents/dbw_figs/'

repeats = 2
graph_names = ['Mouse Connectome', 'Random', 'Small-World', 'Scale-Free']
metrics = [nx.average_clustering, nx.average_shortest_path_length]

met_arr = -1 * np.ones((len(graph_names), repeats, len(metrics)))


def calc_metrics(G, metrics):
    """Helper function to calculate list of (function) metrics"""
    metric_vals = np.zeros((len(metrics)))
    for fi, func in enumerate(metrics):
        metric_vals[fi] = func(G)


###############################
# Create graph/ compute metrics
###############################

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = G_brain.order()

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Calculate metrics specially because no repeats can be done on brain
brain_metrics = calc_metrics(G_brain, metrics)
for met_i, bm in enumerate(metrics):
    met_arr[graph_names.index('Mouse Connectome'), :, met_i] = bm(G_brain)

for rep in np.arange(repeats):
    if 'Amplified Pref. Attachment' in graph_names:
        pass

    if 'Random' in graph_names:
        # Configuration model (random with fixed degree sequence)
        G_CM = nx.random_degree_sequence_graph(sequence=brain_degree,
                                               tries=100)
        met_arr[graph_names.index('Random'), rep, :] = \
            calc_metrics(G_CM, metrics)

    '''
    if 'Small-World' in graph_names:
        # Watts-Strogatz
        G_WS = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)),
                                    0.159)

    if 'Scale-Free' in graph_names:
        # Barabasi-Albert
        G_BA = nx.barabasi_albert_graph(n_nodes,
                                        int(round(brain_degree_mean / 2.)))
        #BA_degree = nx.degree(G_BA).values()
        #BA_clustering = nx.clustering(G_BA).values()
    '''

##########################
# Save metrics in csv file
##########################

#np.savetxt(op.join(save_dir, 'undirected_metrics.csv', met_arr, delimiter=',')
