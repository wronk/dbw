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

import collect_areas
import network_gen

# Network generation parameters
p_th = .01  # P-value threshold
w_th = 0  # Weight-value threshold

# Set relative directory path to linear model & ontology
dir_LM = '../friday-harbor/linear_model'

# Load weights & p-values
W, P, row_labels, col_labels = network_gen.load_weights(dir_LM)
# Threshold weights according to weights & p-values
W_net, mask = network_gen.threshold(W, P, p_th=p_th, w_th=w_th)
# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net == -1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net, 0)
# Put everything in a dictionary
W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
              'data': W_net}
# Convert to networkx graph object
G = network_gen.import_weights_to_graph(W_net_dict)

# Collect & sort areas & edges according to various attributes
sorted_areas = collect_areas.collect_and_sort(G,W_net,labels=row_labels,
                                              print_out=True)
                                              
                                              