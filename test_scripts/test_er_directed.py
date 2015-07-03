from __future__ import print_function, division

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

n = 426
p = 0.04871582435791218
n_edges_brain = 8820

N_TRIALS = 100

n_self_edges = []
n_edges_er = []
for _ in range(N_TRIALS):
    G = nx.erdos_renyi_graph(n=426, p=p, directed=True)
    A = nx.adjacency_matrix(G)

    n_self_edges += [A.diagonal().sum()]
    n_edges_er += [len(G.edges())]

n_graphs_with_self_edges = np.sum(np.array(n_self_edges) > 0)
print('Number of graphs with self edges = {}'.format(n_graphs_with_self_edges))

plt.hist(n_edges_er)
plt.axvline(n_edges_brain, color='k', lw=2)
plt.xlabel('number of edges')
plt.legend(['brain', 'er graphs'])
plt.show()