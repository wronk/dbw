"""
Created on Mon Nov 17 11:06:07 2014

@author: rkp
"""

import extract.brain_graph
import metrics.binary_undirected

import numpy as np
import matplotlib.pyplot as plt

G_brain, A_brain, _, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()

cxn_prob, cxn_bins = metrics.binary_undirected.cxn_probability(A_brain, D_brain)

bin_centers = .5 * (cxn_bins[:-1] + cxn_bins[1:])

plt.bar(bin_centers, cxn_prob, width=.2)