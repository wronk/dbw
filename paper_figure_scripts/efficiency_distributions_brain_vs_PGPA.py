from __future__ import print_function, division
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

from extract import brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as pgpa
from metrics import binary_directed as metrics

import brain_constants as bc
from config import COLORS


# load brain graph
print('loading brain graph...')
G_brain, _, _ = brain_graph.binary_directed()

# create directed ER graph
print('generating directed ER graph...')
G_er = nx.erdos_renyi_graph(bc.num_brain_nodes,
                            bc.p_brain_edge_directed,
                            directed=True)

# create pgpa graph
print('generating model...')
G_pgpa, _, _ = pgpa(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=.75)

print('calculating efficiencies')
# calculate local efficiency distribution
for ctr, G in enumerate((G_brain, G_er, G_pgpa)):
    print('for graph {}'.format(ctr))
    G.efficiency = metrics.efficiency_matrix(G)

# plot histograms of all three efficiency matrices
fig, axs = plt.subplots(1, 2)

print('making histograms')
axs[0].hist([G_brain.efficiency[G_brain.efficiency >= 0],
             G_er.efficiency[G_er.efficiency >= 0],
             G_pgpa.efficiency[G_pgpa.efficiency >= 0]],
            color=(COLORS['brain'], COLORS['er'], COLORS['pgpa']))

axs[1].hist([G_brain.efficiency.mean(axis=1),
             G_er.efficiency.mean(axis=1),
             G_pgpa.efficiency.mean(axis=1)],
            color=(COLORS['brain'], COLORS['er'], COLORS['pgpa']))

plt.draw()
plt.show(block=True)
