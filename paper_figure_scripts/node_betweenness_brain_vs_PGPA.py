"""
Calculate the node betweenness distributions for the brain and the PGPA model (the latter averaged over several instantiations).
"""

from __future__ import print_function, division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

from extract import brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa
from network_plot import change_settings

import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE
from config import COLORS, FACE_COLOR, AX_COLOR, FONT_SIZE

# parameters for this particular plot
FIG_SIZE = (12, 6)
N_GRAPH_SAMPLES = 100
BINS = np.linspace(0, .03, 25)
BINCS = 0.5 * (BINS[:-1] + BINS[1:])

# load brain graph
print('loading brain graph...')
G_brain, _, _ = brain_graph.binary_directed()

print('looping over {} graph instantiations...'.format(N_GRAPH_SAMPLES))

graphs_er = []
graphs_pgpa = []

for g_ctr in range(N_GRAPH_SAMPLES):
    if (g_ctr + 1) % 5 == 0:
        print('{} of {} samples completed.'.format(g_ctr + 1, N_GRAPH_SAMPLES))

    # create directed ER graph
    G_er = nx.erdos_renyi_graph(bc.num_brain_nodes,
                                bc.p_brain_edge_directed,
                                directed=True)

    # create pgpa graph
    G_pgpa, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=LENGTH_SCALE)

    # calculate betweenness distribution for G_er and G_pgpa
    for label, G in zip(['er', 'pgpa'], [G_er, G_pgpa]):
        betweenness = nx.betweenness_centrality(G).values()
        G.counts_betweenness = np.histogram(betweenness, bins=BINS)[0]
        if label == 'er':
            graphs_er += [G]
        elif label == 'pgpa':
            graphs_pgpa += [G]

# calculate mean and std of betweenness counts for er and pgpa
counts_betweenness_mean_er = np.array([G.counts_betweenness for G in graphs_er]).mean(axis=0)
counts_betweenness_std_er = np.array([G.counts_betweenness for G in graphs_er]).std(axis=0)
counts_betweenness_mean_pgpa = np.array([G.counts_betweenness for G in graphs_pgpa]).mean(axis=0)
counts_betweenness_std_pgpa = np.array([G.counts_betweenness for G in graphs_pgpa]).std(axis=0)

# plot histograms of all three betweenness distributions
fig, ax = plt.subplots(1, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, tight_layout=True)

# brain
ax.hist(nx.betweenness_centrality(G_brain).values(), bins=BINS, color=COLORS['brain'], lw=0)

# er
ax.plot(BINCS, counts_betweenness_mean_er, color=COLORS['er'], lw=2)
ax.fill_between(BINCS, counts_betweenness_mean_er - counts_betweenness_std_er,
                counts_betweenness_mean_er + counts_betweenness_std_er,
                color=COLORS['er'], alpha=0.3)

# pgpa
ax.plot(BINCS, counts_betweenness_mean_pgpa, color=COLORS['pgpa'], lw=2)
ax.fill_between(BINCS, counts_betweenness_mean_pgpa - counts_betweenness_std_pgpa,
                counts_betweenness_mean_pgpa + counts_betweenness_std_pgpa,
                color=COLORS['pgpa'], alpha=0.3)

ax.set_xlim(0, 0.03)
ax.set_ylim(0, 250)

ax.set_xlabel('node betweenness')
ax.set_ylabel('counts')

change_settings.set_all_colors(ax, AX_COLOR)
change_settings.set_all_text_fontsizes(ax, FONT_SIZE)

plt.draw()
plt.show(block=True)