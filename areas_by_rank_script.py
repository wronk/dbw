import pdb
"""
Created on Wed Aug 27 23:17:01 2014

@author: rkp

Analyze properties of specific brain areas with extreme ranks according to 
specific criteria.
"""

import pprint as pp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import network_gen
import network_compute

from friday_harbor.structure import Ontology

# Network generation parameters
p_th = .01 # P-value threshold
w_th = 0 # Weight-value threshold

# Analysis parameters
top_out_in = 10
bot_out_in = 10
top_node_btwn = 10
top_edge_btwn = 20
top_degree = 10
top_ccoeff = 10

# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'
dir_onto = '../data'

# Get ontological dictionary with acronyms as keys
onto = Ontology(data_dir=dir_onto)
onto_dict = {s.acronym:s.name for s in onto.structure_list}

# Load weights & p-values
W,P,row_labels,col_labels = network_gen.load_weights(dir_LM)
# Threshold weights according to weights & p-values
W_net,mask = network_gen.threshold(W,P,p_th=p_th,w_th=w_th)
# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net==-1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net,0)

# Put everything in a dictionary
W_net_dict = {'row_labels':row_labels,'col_labels':col_labels,
              'data':W_net}

# Convert to networkx graph object
G = network_gen.import_weights_to_graph(W_net_dict) 

# Collect nodes ranked according to output/input ratio
out_in_dict = network_compute.out_in_ratio(W_net,labels=row_labels)
out_in_labels,out_in_vals = network_compute.get_ranked(out_in_dict)
# Get names
out_in_names = ['%s_%s'%(onto_dict[label[:-2]],label[-2:]) \
    for label in out_in_labels]
# Print out top-ranked nodes
print 'Top nodes by output/input ratio:'
top_out_in_tuples = zip(out_in_labels[:top_out_in],out_in_names[:top_out_in],out_in_vals[:top_out_in])
pp.pprint(top_out_in_tuples)

# Print out bottom-ranked nodes
print 'Bottom nodes by output/input ratio:'
bot_out_in_tuples = zip(out_in_labels[-bot_out_in:],out_in_names[-bot_out_in:],out_in_vals[-bot_out_in:])
pp.pprint(bot_out_in_tuples)


# Collect nodes ranked according to node-betweenness
node_btwn_dict = nx.betweenness_centrality(G)
node_btwn_labels,node_btwn_vals = network_compute.get_ranked(node_btwn_dict)
# Get names
node_btwn_names = ['%s_%s'%(onto_dict[label[:-2]],label[-2:]) \
    for label in node_btwn_labels]
# Print out top-ranked nodes
print 'Top nodes by node-betweenness:'
top_node_btwn_tuples = zip(node_btwn_labels[:top_node_btwn],node_btwn_names[:top_node_btwn],node_btwn_vals[:top_node_btwn])
pp.pprint(top_node_btwn_tuples)

# Collect nodes ranked according to degree distribution
degree_dict = G.degree()
degree_labels,degree_vals = network_compute.get_ranked(degree_dict)
# Get names
degree_names = ['%s_%s'%(onto_dict[label[:-2]],label[-2:]) \
    for label in degree_labels]
# Print out top-ranked nodes
print 'Top nodes by degree:'
top_degree_tuples = zip(degree_labels[:top_degree],degree_names[:top_degree],degree_vals[:top_degree])
pp.pprint(top_degree_tuples)

# Collect nodes ranked according to clustering coefficient
ccoeff_dict = nx.clustering(G)
ccoeff_labels,ccoeff_vals = network_compute.get_ranked(ccoeff_dict)
# Get names
ccoeff_names = ['%s_%s'%(onto_dict[label[:-2]],label[-2:]) \
    for label in ccoeff_labels]
# Print out top-ranked nodes
print 'Top nodes by clustering coefficient:'
top_ccoeff_tuples = zip(ccoeff_labels[:top_ccoeff],ccoeff_names[:top_ccoeff],ccoeff_vals[:top_ccoeff])
pp.pprint(top_ccoeff_tuples)