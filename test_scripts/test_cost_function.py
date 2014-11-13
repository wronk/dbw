"""
Created on Wed Aug 27 23:16:17 2014

@author: rkp

Test creation of random weighted undirected graphs.
"""

import numpy as np
import extract.brain_graph
import metrics.weighted_undirected

# Get mouse connectivity graph & distance
G, W, _, _ = extract.brain_graph.weighted_undirected()
D, _ = extract.brain_graph.distance_matrix()

# Calculate swapped-cost distribution for graph
cost_change_dist = metrics.weighted_undirected.swapped_cost_distr(W,D)