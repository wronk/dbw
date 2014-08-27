import pdb
"""
Created on Wed Aug 27 16:06:11 2014

@author: rkp

Script for plotting things & printing values for control network.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import network_gen
import rich_plot as plot_net_properties

# Set parameters
p_th = .01 # P-value threshold
w_th = 0 # Weight-value threshold

# Set relative directory path
dir_name = '../friday-harbor/linear_model'

# Load weights & p-values
W,P,row_labels,col_labels = network_gen.load_weights(dir_name)
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

# Plot things
plot_net_properties.plot_node_btwn(G) # Node-betweenness
plot_net_properties.plot_edge_btwn(G) # Edge-betweenness
plot_net_properties.plot_out_in_ratios(W_net,labels=row_labels)