from __future__ import print_function, division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

from extract import brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa
from random_graph.binary_directed import biophysical_indegree as pa
from network_plot import change_settings

import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE

from config import COLORS, FACE_COLOR, AX_COLOR, FONT_SIZE

# parameters for this particular plot
FACE_COLOR = 'w'
ALPHA = .6
FIG_SIZE = (12, 6)

# load brain graph
print('loading brain graph...')
G_brain, _, _ = brain_graph.binary_directed()

# create pgpa graphs
print('generating model...')
G_pgpa_L_inf, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=np.inf)

print('generating model...')
G_pgpa, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=LENGTH_SCALE)

print('generating model...')
G_pa, _, _ = pa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=np.inf)

# cast both models to undirected graphs

# plot preferential growth (L = np.inf) and pref attachment
fig, ax = plt.subplots(1, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE,
                       tight_layout=True)

colors = (COLORS['pref-attachment'], COLORS['pref-growth'], COLORS['pgpa'], COLORS['brain'])

for G, color in zip((G_pa, G_pgpa_L_inf, G_pgpa, G_brain), colors):
    deg = nx.degree(G.to_undirected()).values()
    cc = nx.clustering(G.to_undirected()).values()
    ax.scatter(deg, cc, c=color, lw=0, alpha=ALPHA)

ax.set_xlabel('Degree')
ax.set_ylabel('Clustering\ncoefficient')

ax.set_xlim(0, 150)
ax.set_xticks((0, 50, 100, 150))
ax.set_ylim(0, 1)
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))

ax.legend(('preferential attachment', 'preferential growth', 'PGPA', 'brain'), fontsize=FONT_SIZE)

change_settings.set_all_text_fontsizes(ax, FONT_SIZE)

plt.draw()
plt.show(block=True)