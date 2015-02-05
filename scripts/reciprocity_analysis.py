"""
Created on Fri Jan 30 16:53:44 2015

@author: rkp

Analyze reciprocity of brain & model graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree_reciprocal as model_graph
from metrics.binary_directed import reciprocity
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

from brain_constants import *

# PARAMETERS
NER = 100 # number of ER-directed graphs to generate
NMODEL = 100

# load brain graph, adjacency matrix, and labels
Gbrain, Abrain, labels = brain_graph()
reciprocity_brain = reciprocity(G)

# calculate reciprocity for several ER-directed graphs
reciprocity_ER = np.zeros((NER,), dtype=float)
for ctr in range(NER):
    GER = nx.erdos_renyi_graph(num_brain_nodes, p_brain_edge_directed, 
                               directed=True)
    reciprocity_ER[ctr] = reciprocity(GER)

# calculate reciprocity for several model graphs
reciprocity_model = np.zeros((NMODEL,), dtype=float)
for ctr in range(NMODEL):
    print ctr
    Gmodel = model_graph(N=num_brain_nodes, 
                         N_edges=num_brain_edges_directed, 
                         gamma=1., reciprocity=7.)
    reciprocity_model[ctr] = reciprocity(Gmodel)

# plot histograms
fig, ax = plt.subplots(1, 1, facecolor='white', tight_layout=True)
ax.hist(reciprocity_ER, lw=0)
ax.hist(reciprocity_model, lw=0)
ax.axvline(reciprocity_brain, lw=3)
set_fontsize(ax, fontsize=20)