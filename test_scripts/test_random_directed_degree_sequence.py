from __future__ import division, print_function

N_TRIALS = 100

import matplotlib.pyplot as plt

from random_graph import binary_directed
from extract import brain_graph


G_brain, A_brain, _ = brain_graph.binary_directed()

in_sequence = G_brain.in_degree().values()
out_sequence = G_brain.out_degree().values()

n_edges_random = []
for _ in range(N_TRIALS):
    G, A, _ = binary_directed.random_directed_deg_seq(in_sequence,
                                                      out_sequence,
                                                      simplify=True)
    n_edges_random += [len(G.edges())]

_, ax = plt.subplots(1, 1)
ax.hist(n_edges_random, color='b')
ax.axvline(len(G_brain.edges()), color='k')
ax.legend(['random', 'brain'])
ax.set_xlabel('number of edges in graph')
plt.show(block=True)