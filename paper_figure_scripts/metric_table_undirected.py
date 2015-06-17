"""
Created Sat May 23 14:11:50 2015

@author: wronk

Create a table of useful metrics for undirected standard graphs and model
"""

import os
import os.path as op
import numpy as np
import networkx as nx

import extract.brain_graph
from random_graph import binary_undirected
from random_graph.binary_undirected import \
    undirected_biophysical_reverse_outdegree as pgpa_und
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa_dir
from metrics import binary_undirected as und_metrics
reload(und_metrics)
import pickle
import brain_constants as bc

from config.graph_parameters import LENGTH_SCALE


def calc_metrics(G, metrics):
    """Helper function to calculate list of (function) metrics"""
    metric_vals = np.zeros((len(metrics)))
    for fi, func in enumerate(metrics):
        metric_vals[fi] = func(G)

    return metric_vals

###############################
# Parameters
###############################

# SET YOUR SAVE DIRECTORY
save_dir = os.environ['DBW_SAVE_CACHE']

repeats = 100

# Set the graphs and metrics you wisht to include
graph_names = ['Mouse Connectome', 'Random', 'Small-World', 'Scale-Free',
               'PGPA']
#graph_names = ['Mouse Connectome', 'Small-World']
#metrics = [und_metrics.local_efficiency, und_metrics.global_efficiency]
metrics = [nx.average_clustering, nx.average_shortest_path_length,
           und_metrics.global_efficiency, und_metrics.local_efficiency]
brain_size = [7., 7., 7.]

###############################
# Create graph/ compute metrics
###############################
#TODO: consider using a dict instead of checking for each graph type

# Initialize matrix to store metric values
met_arr = -1 * np.ones((len(graph_names), repeats, len(metrics)))

# Load mouse connectivity graph
G_brain, _, _ = extract.brain_graph.binary_undirected()
n_nodes = G_brain.number_of_nodes()
n_edges = G_brain.number_of_edges()
G_brain_copy = G_brain.copy()

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Calculate metrics specially because no repeats can be done on brain
brain_metrics = calc_metrics(G_brain, metrics)
for met_i, bm in enumerate(metrics):
    met_arr[graph_names.index('Mouse Connectome'), :, met_i] = bm(G_brain)

for rep in np.arange(repeats):
    # PGPA model
    if 'PGPA' in graph_names:
        G_PGPA, _, _ = pgpa_dir(bc.num_brain_nodes,
                                L=LENGTH_SCALE)
        met_arr[graph_names.index('PGPA'), rep, :] = \
            calc_metrics(G_PGPA.to_undirected(), metrics)

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

    print 'Completed repeat: ' + str(rep)

##########################
# Save metrics
##########################
save_dict = dict(met_arr=met_arr, metrics=metrics, graph_names=graph_names)

# Save original array/information and csv version averaged across repeats
f_name = op.join(save_dir, 'undirected_metrics')

# Save all original info in pickled dict (for potential plotting later)
outfile = open(f_name + '.pkl', 'wb')
pickle.dump(save_dict, outfile)
outfile.close()

# Save csvs for easy table creation
np.savetxt(f_name + '_averaged.csv', met_arr.mean(1), fmt='%10.3f',
           delimiter=',')
