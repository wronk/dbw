import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: rkp

Analyze properties of specific brain areas with extreme ranks according to
specific criteria.
"""

import numpy as np
import networkx as nx

import print_net
import network_compute

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

# Analysis parameters
top_out_in = 10
bot_out_in = 10
top_node_btwn = 10
top_edge_btwn = top_node_btwn*(top_node_btwn-1)/2
top_degree = 10
top_ccoeff = 10

def collect_and_sort(G,W_net,labels,print_out=True):
    """Collect lists of nodes & edges sorted by certain attributes.
    
    Returns dictionary of node & edge labels sorted according to specific
    criteria, which are available alongside the sorted labels."""
    # Collect nodes ranked according to degree distribution
    degree_dict = G.degree()
    degree_labels, degree_vals = network_compute.get_ranked(degree_dict)
    
    # Collect nodes ranked according to output/input ratio
    out_in_dict = network_compute.out_in_ratio(W_net, labels=labels)
    out_in_labels, out_in_vals = network_compute.get_ranked(out_in_dict)
    
    # Collect nodes ranked according to clustering coefficient
    ccoeff_dict = nx.clustering(G)
    ccoeff_labels, ccoeff_vals = network_compute.get_ranked(ccoeff_dict)
    
    # Collect nodes ranked according to node-betweenness
    node_btwn_dict = nx.betweenness_centrality(G)
    node_btwn_labels, node_btwn_vals = network_compute.get_ranked(node_btwn_dict)
    
    # Collect edges ranked by edge-betweenness
    edge_btwn_dict = nx.edge_betweenness_centrality(G)
    edge_btwn_labels, edge_btwn_vals = network_compute.get_ranked(edge_btwn_dict)
    
    if print_out:
        print 'By degree:'
        print_net.print_node_list(degree_labels,degree_vals,top_degree)
        print 'By output/input (top):'
        print_net.print_node_list(out_in_labels,out_in_vals,top_out_in)
        print 'By output/input (bottom):'
        print_net.print_node_list(out_in_labels,out_in_vals,-bot_out_in)
        print 'By clustering coefficient:'
        print_net.print_node_list(ccoeff_labels,ccoeff_vals,top_ccoeff)
        print 'By node-betweenness:'
        print_net.print_node_list(node_btwn_labels,node_btwn_vals,top_node_btwn)
        print 'By edge-betweenness:'
        print_net.print_edge_list(edge_btwn_labels,edge_btwn_vals,top_edge_btwn)
            # Print out what percent of the top edge-betweenness edges touch the top
        # node-betweenness nodes & what percent of them connect two top node-
        # betweenness nodes
        edges_touching, edges_connecting = \
            network_compute.node_edge_overlap(node_btwn_labels[:top_node_btwn],
                                              edge_btwn_labels[:top_edge_btwn])
        print 'Top edges touching top nodes (by node-betweenness):'
        print '%d/%d'%(len(edges_touching),top_edge_btwn)
        
        print 'Top edges connecting top nodes (by node-betweenness):'
        print '%d/%d'%(len(edges_connecting),top_edge_btwn)
        
        # Print out what percent of the top edge-betweenness edges touch the top
        # degree nodes & what percent of them connect two top degree nodes
        edges_touching, edges_connecting = \
            network_compute.node_edge_overlap(degree_labels[:top_degree],
                                              edge_btwn_labels[:top_edge_btwn])
        print 'Top edges touching top nodes (by degree):'
        print '%d/%d'%(len(edges_touching),top_edge_btwn)
        
        print 'Top edges connecting top nodes (by degree):'
        print '%d/%d'%(len(edges_connecting),top_edge_btwn)
        
    sorted_dict = {'degree_labels':degree_labels,
                   'degree_vals':degree_vals,
                   'out_in_labels':out_in_labels,
                   'out_in_vals':out_in_vals,
                   'ccoeff_labels':ccoeff_labels,
                   'ccoeff_vals':ccoeff_vals,
                   'node_btwn_labels':node_btwn_labels,
                   'node_btwn_vals':node_btwn_vals,
                   'edge_btwn_labels':edge_btwn_labels,
                   'edge_btwn_vals':edge_btwn_vals}
                   
    return sorted_dict