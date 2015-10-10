"""
Create and save percolation data for standard, brain, and model graphs.
"""

import os
from os import path as op
import numpy as np
import networkx as nx
import pickle

from extract import brain_graph
from metrics import percolation as perc
reload(perc)
import brain_constants as bc
from random_graph.binary_directed import biophysical_reverse_outdegree as \
    pgpa_dir
from metrics import binary_undirected as und_metrics
from random_graph import binary_undirected as und_graphs

from config.graph_parameters import LENGTH_SCALE

repeats = 100  # Number of times to repeat percolation
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
node_order = 426
brain_size = [7., 7., 7.]

save_dir = os.environ['DBW_SAVE_CACHE']
#data_dir = os.environ['DBW_DATA_DIRECTORY']
save_files = True
gen_directed = False
##############################################################################


def construct_graph_list_und(graphs_to_const):
    """Construct and return a list of graphs so graph construction is easily
    repeatable"""

    graph_check = ['Random', 'Small-world', 'Scale-free', 'SGPA']
    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_brain, _, _ = brain_graph.binary_undirected()
    graph_list.append(G_brain)

    # Calculate degree & clustering coefficient distribution
    n_nodes = G_brain.order()

    brain_degree = nx.degree(G_brain).values()
    brain_degree_mean = np.mean(brain_degree)

    # Construct degree controlled random
    if graph_check[0] in graphs_to_const:
        G_RAND, _, _ = und_graphs.random_simple_deg_seq(
            sequence=brain_degree, brain_size=brain_size, tries=100)
        graph_list.append(G_RAND)

    # Construct small-world graph
    if graph_check[1] in graphs_to_const:
        graph_list.append(nx.watts_strogatz_graph(
            n_nodes, int(round(brain_degree_mean)), 0.159))

    # Construct scale-free graph
    if graph_check[2] in graphs_to_const:
        graph_list.append(nx.barabasi_albert_graph(
            n_nodes, int(round(brain_degree_mean / 2.))))

    # Construct SGPA graph
    if graph_check[3] in graphs_to_const:
        G_SGPA, _, _ = pgpa_dir(bc.num_brain_nodes,
                                bc.num_brain_edges_directed,
                                L=LENGTH_SCALE)
        graph_list.append(G_SGPA.to_undirected())

    # Error check that we created correct number of graphs
    assert len(graph_list) == len(graphs_to_const), (
        'Graph list/names don\'t match')

    return graph_list


'''
def construct_graph_list_dir(graphs_to_const):
    """Construct and return a list of directed graphs so graph construction
    is easily repeatable"""

    graph_check = ['Random', 'Biophysical']
    #graph_check = ['Random', 'Small-World', 'Scale-Free', 'Biophysical']
    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_AL, _, _ = brain_graph.und_graphs()
    graph_list.append(G_AL)

    # Calculate degree & clustering coefficient distribution
    n_nodes = G_AL.order()
    #n_edges = float(len(G_AL.edges()))
    #p_edge = n_edges / ((n_nodes * (n_nodes - 1)) / 2.)

    brain_degree = nx.degree(G_AL).values()
    brain_degree_mean = np.mean(brain_degree)

    # Construct Random (ER) graph
    if graph_check[0] in graphs_to_const:
        graph_list.append(nx.erdos_renyi_graph(bc.num_brain_nodes,
                                               bc.p_brain_edge_directed,
                                               directed=True))

    # Construct pgpa graph
    if graph_check[1] in graphs_to_const:
        G_SGPA, _, _ = pgpa(bc.num_brain_nodes, bc.num_brain_edges_directed,
                            L=LENGTH_SCALE)
        graph_list.append(G_SGPA)

    # Error check that we created correct number of graphs
    assert (len(graph_list) == len(graphs_to_const),
            'Graph list/names don\'t match')

    return graph_list
'''

##################
# Construct graphs
##################

if gen_directed:
    graph_names = ['Connectome', 'Random', 'SGPA']
else:
    graph_names = ['Connectome', 'Random', 'Small-world', 'Scale-free',
                   'SGPA']

# Directed
if gen_directed:
    func_list = [(perc.lesion_met_largest_strong_component,
                  'Largest Strong Component'),
                 (perc.lesion_met_avg_shortest_path,
                  'Average Shortest Path'),
                 (perc.lesion_met_largest_weak_component,
                  'Largest Weak Componenet')]

# Undirected
else:
    func_list = [(perc.lesion_met_largest_component, 'Largest component'),
                 (und_metrics.global_efficiency, 'Global efficiency')]

#################
# Do percolation
#################

print 'Building percolation data...'
print 'Graphs: ' + str(graph_names)
print 'Directed: ' + str(gen_directed) + '\n'
if not save_files and repeats > 10:
    '\nAre you sure you don\'t want to save results?\n'
# Matrices for random and targetted attacks
rand = np.zeros((len(graph_names), len(func_list), len(prop_rm), repeats))
targ = np.zeros((len(graph_names), len(func_list), len(lesion_list), repeats))

# Percolation
for ri in np.arange(repeats):
    print 'Lesioning; repeat ' + str(ri + 1) + ' of ' + str(repeats)
    '''
    # Construct undirected or directed graphs
    if gen_directed:
        graph_list = construct_graph_list_dir(graph_names)
    else:
        graph_list = construct_graph_list_und(graph_names)
        '''
    graph_list = construct_graph_list_und(graph_names)

    # Cycle over graphs and metric functions
    for gi, G in enumerate(graph_list):
        print '\tLesioning: ' + graph_names[gi],
        for fi, (func, func_label) in enumerate(func_list):
            rand[gi, fi, :, ri] = perc.percolate_random(G.copy(), prop_rm,
                                                        func)
            targ[gi, fi, :, ri] = perc.percolate_degree(G.copy(), lesion_list,
                                                        func)
        print ' ... Done'

###############
# Save results
###############

if save_files:
    print 'Saving data for: '
    for gi, G in enumerate(graph_list):
        print '\tSaving: ' + graph_names[gi],

        '''
        if gen_directed:
            save_fname = op.join(save_dir, graph_names[gi] + '_directed.pkl')
        else:
            save_fname = op.join(save_dir, graph_names[gi] + '_undirected.pkl')
        '''
        save_fname = op.join(save_dir, graph_names[gi] +
                             '_undirected_perc.pkl')

        outfile = open(save_fname, 'wb')
        pickle.dump({'graph_name': graph_names[gi], 'metrics_list':
                    [f.func_name for (f, f_label) in func_list],
                    'repeats': repeats,
                    'metrics_label': [f_label for (f, f_label) in func_list],
                    'data_rand': rand[gi, :, :, :],
                    'data_targ': targ[gi, :, :, :],
                    'removed_rand': prop_rm,
                    'removed_targ': lesion_list}, outfile)
        outfile.close()

        print ' ... Done'
