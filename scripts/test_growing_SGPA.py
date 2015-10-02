from __future__ import division
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
import numpy as np
import networkx as nx

from random_graph import binary_directed as bd

N_NODES = 426
P_EDGE_SPLIT = .013
L = 0.7
BRAIN_SIZE = [7, 7, 7]

G = bd.growing_SGPA_1(N_NODES, P_EDGE_SPLIT, L, BRAIN_SIZE, remove_extra_ccs=True)

print 'connected components'
for cc in nx.connected_components(G.to_undirected()):
    print len(cc)

in_deg = np.array(G.in_degree().values())
out_deg = np.array(G.out_degree().values())
total_deg = in_deg + out_deg

fig, axs = plt.subplots(1, 4)
axs[0].scatter(in_deg, out_deg)
axs[0].set_xlabel('in-degree')
axs[0].set_ylabel('out-degree')
axs[1].scatter(total_deg, in_deg/total_deg)
axs[1].set_xlabel('total degree')
axs[1].set_ylabel('prop in-degree')

axs[2].hist(nx.degree(G.to_undirected()).values(), bins=30)
axs[2].set_xlabel('degree')
axs[2].set_ylabel('number of nodes')

axs[3].scatter(
    nx.degree(G.to_undirected()).values(), nx.clustering(G.to_undirected()).values(),
    c=np.array(nx.degree(G.to_undirected()).keys())/N_NODES, cmap=cm.jet
)
axs[3].set_xlabel('degree')
axs[3].set_ylabel('clustering')
plt.show(block=True)