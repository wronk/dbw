from __future__ import division, print_function

N_TRIALS = 100

import matplotlib.pyplot as plt

from random_graph import binary_directed
from extract import brain_graph
import networkx as nx

G_brain, _, _ = brain_graph.binary_directed()
p = G_brain.number_of_nodes() / G_brain.number_of_edges()

in_sequence = G_brain.in_degree().values()
out_sequence = G_brain.out_degree().values()

labels = ['Deg. Cont. Rand.', 'ER Rand.', 'Brain']
n_edges_deg_controlled_random = []
n_edges_ER_random = []

for ri in range(N_TRIALS):
    G_RAND, _, _ = binary_directed.random_directed_deg_seq(in_sequence,
                                                           out_sequence,
                                                           simplify=True)
    G_ER, _, _ = binary_directed.ER_distance(p=p)
    G_ER = nx.DiGraph(G_ER)  # Remove parallel
    G_ER.remove_edges_from(G_ER.selfloop_edges())  # Remove self-loops

    n_edges_deg_controlled_random.append(G_RAND.number_of_edges())
    n_edges_ER_random.append(G_ER.number_of_edges())
    print('Finished repeat: ' + str(ri + 1) + ' of ' + str(N_TRIALS))

_, ax = plt.subplots(1, 1)
ax.hist(n_edges_deg_controlled_random, color='b', alpha=0.5, label=labels[0])
ax.hist(n_edges_ER_random, color='r', alpha=0.5, label=labels[1])
ax.axvline(len(G_brain.edges()), color='k', label=labels[2])
#ax.legend(['Deg. Cont. Rand', 'ER Rand.', 'Brain'])
ax.legend(loc='best')
ax.set_xlabel('number of edges in graph')
plt.show(block=True)
