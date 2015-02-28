"""
Created on Wed Feb 25 11:29:20 2015

@author: wronk

Create and save percolation data for standard and brain graphs.
"""

import os
from os import path as op
import numpy as np
import networkx as nx
import pickle

from extract import brain_graph
#import network_gen as ng
from metrics import percolation as perc
reload(perc)
import brain_constants as bc
from random_graph.binary_undirected import undirected_biophysical_reverse_outdegree as bp_und
from random_graph.binary_directed import biophysical_reverse_outdegree as bp_dir
import in_out_plot_config as cf

repeats = 100  # Number of times to repeat percolation
prop_rm = np.arange(0., 1.00, 0.05)
lesion_list = np.arange(0, 426, 10)
node_order = 426

linear_model_path = op.join('home', os.environ['USER'],
                            'Builds/friday-harbor/linear_model')
save_dir = op.join('/home', os.environ['USER'], 'Builds/dbw/cache')
save_files = True
gen_directed = True
##############################################################################


def construct_graph_list_und(graphs_to_const):
    ''' Construct and return a list of graphs so graph construction is easily
    repeatable'''

    graph_check = ['Random', 'Small-World', 'Scale-Free', 'Biophysical']
    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_AL, _, _ = brain_graph.binary_undirected()
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
                                               bc.p_brain_edge_undirected))

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

        G_BIO, _, _ = bp_und(bc.num_brain_nodes, bc.num_brain_edges_undirected,
                             L=0.75, gamma=1.)
        graph_list.append(G_BIO)

    # Error check that we created correct number of graphs
    assert len(graph_list) == len(graphs_to_const), 'Graph list/names don\'t match'

    return graph_list


def construct_graph_list_dir(graphs_to_const):
    ''' Construct and return a list of directed graphs so graph construction
    is easily repeatable'''

    graph_check = ['Random', 'Biophysical']
    #graph_check = ['Random', 'Small-World', 'Scale-Free', 'Biophysical']
    graph_list = []

    # Always construct and add Allen Institute mouse brain to list
    G_AL, _, _ = brain_graph.binary_directed()
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

    # Construct biophysical graph
    if graph_check[1] in graphs_to_const:
        G_BIO, _, _ = bp_dir(N=bc.num_brain_nodes,
                             N_edges=bc.num_brain_edges_directed, L=.75,
                             gamma=1.)
        graph_list.append(G_BIO)
    '''
    # Construct WS graph
    if graph_check[2] in graphs_to_const:
        graph_list.append(nx.watts_strogatz_graph(
            n_nodes, int(round(brain_degree_mean)), 0.159))

    # Construct BA graph
    if graph_check[3] in graphs_to_const:
        graph_list.append(nx.barabasi_albert_graph(
            n_nodes, int(round(brain_degree_mean / 2.))))
            '''

    # Error check that we created correct number of graphs
    assert len(graph_list) == len(graphs_to_const), 'Graph list/names don\'t match'

    return graph_list

##############################################################################
### Construct graphs
if gen_directed:
    graph_names = ['Mouse', 'Random', 'Biophysical']
else:
    graph_names = ['Mouse', 'Random', 'Small-World', 'Scale-Free',
                   'Biophysical']

# Directed
if gen_directed:
    func_list = [(perc.lesion_met_largest_strong_component, 'Largest Strong Component'),
                 (perc.lesion_met_avg_shortest_path, 'Average Shortest Path'),
                 (perc.lesion_met_largest_weak_component, 'Largest Weak Componenet')]

# Undirected
else:
    func_list = [(perc.lesion_met_largest_component, 'Largest Component'),
                 (perc.lesion_met_avg_shortest_path, 'Average Shortest Path')]

##############################################################################
### Do percolation
print 'Building percolation data...'
print 'Graphs: ' + str(graph_names)
print 'Directed: ' + str(gen_directed) + '\n'
# Matrices for random and targetted attacks
rand = np.zeros((len(graph_names), len(func_list), len(prop_rm), repeats))
targ = np.zeros((len(graph_names), len(func_list), len(lesion_list), repeats))

# Percolation
for ri in np.arange(repeats):
    print 'Lesion graphs, repeat ' + str(ri + 1) + ' of ' + str(repeats)
    # Construct undirected or directed graphs
    if gen_directed:
        graph_list = construct_graph_list_dir(graph_names)
    else:
        graph_list = construct_graph_list_und(graph_names)

    # Cycle over graphs and metric functions
    for gi, G in enumerate(graph_list):
        print '\tLesioning: ' + graph_names[gi],
        for fi, (func, func_label) in enumerate(func_list):
            rand[gi, fi, :, ri] = perc.percolate_random(G, prop_rm, func)
            targ[gi, fi, :, ri] = perc.percolate_degree(G, lesion_list, func)
        print ' ... Done'

##############################################################################
### Save results
if save_files:
    print 'Saving data for: '
    for gi, G in enumerate(graph_list):
        print '\tLesioning: ' + graph_names[gi]

        if gen_directed:
            save_fname = op.join(save_dir, graph_names[gi] + '_directed.pkl')
        else:
            save_fname = op.join(save_dir, graph_names[gi] + '_undirected.pkl')

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
        #f.close()
