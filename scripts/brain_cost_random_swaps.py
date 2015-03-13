"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from extract.brain_graph import binary_directed as brain_graph
from graph_tools.auxiliary import swap_node_positions as swap

plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'black'
MARKERCOLOR='m'
FONTSIZE = 24
NBINS = 15

# load brain graph, adjacency matrix, and labels
G, A, D, labels = brain_graph(p_th=.01)

brain_avg_shortest_path = nx.average_shortest_path_length(G, weight='distance')

# randomly swap two nodes
Gswapped, Dswapped = swap(G, D)
swapped_avg_shortest_path = nx.average_shortest_path_length(Gswapped, weight='distance')