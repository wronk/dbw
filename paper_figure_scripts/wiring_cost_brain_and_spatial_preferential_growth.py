"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. out-degree distribution for the reverse outdegree model.
"""

import numpy as np
import matplotlib.pyplot as plt

from extract.brain_graph import binary_directed_with_distance as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from random_graph.binary_directed import ER_distance

from metrics.binary_undirected import swapped_cost_distr

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

# IMPORT PLOT PARAMETERS

plt.ion()

NTRIALS = 500

# PLOT PARAMETERS
FIGSIZE = (11, 4)
FACECOLOR = 'k'
BRAINCOLOR = 'm'
MODELCOLOR = 'orange'
ERCOLOR = 'r'
FONTSIZE = 16
NBINS = 15

# load brain graph
Gbrain, Abrain, Dbrain, labels = brain_graph(p_th=.05)

# create model graph
G, A, D = biophysical_model(N=bc.num_brain_nodes,
                            N_edges=bc.num_brain_edges_directed,
                            L=.75,
                            gamma=1.)

# create random graph
Grandom, Arandom, Drandom = ER_distance(bc.num_brain_nodes,
                                        p=bc.p_brain_edge_directed)

# calculate random swap cost distribution for brain and model
brain_cost = swapped_cost_distr(Abrain, Dbrain, n_trials=NTRIALS)
model_cost = swapped_cost_distr(A, D, n_trials=NTRIALS)
random_cost = swapped_cost_distr(Arandom, Drandom, n_trials=NTRIALS)

# get histograms
brain_cts, brain_bins = np.histogram(brain_cost, NBINS)
model_cts, model_bins = np.histogram(model_cost, NBINS)
random_cts, random_bins = np.histogram(random_cost, NBINS)

# get bin centers
brain_bincs = .5 * (brain_bins[:-1] + brain_bins[1:])
model_bincs = .5 * (model_bins[:-1] + model_bins[1:])
random_bincs = .5 * (random_bins[:-1] + random_bins[1:])

fig, ax = plt.subplots(1, 1, facecolor=FACECOLOR)
ax.vlines(0, 0, 100)

ax.plot(brain_bincs, brain_cts, color=BRAINCOLOR, lw=2)
ax.plot(model_bincs, model_cts, color=MODELCOLOR, lw=2)
ax.plot(random_bincs, random_cts, color=ERCOLOR, lw=2)

ax.set_ylim(0, max(brain_cts.max(), model_cts.max(), random_cts.max()))
ax.set_xlabel('% change in cost')
ax.set_ylabel('counts')

set_all_colors(ax, 'w')
set_all_text_fontsizes(ax, FONTSIZE)
