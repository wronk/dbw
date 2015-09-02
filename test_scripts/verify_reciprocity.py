from __future__ import division
import numpy as np
import networkx as nx
import os

from extract.brain_graph import binary_directed as brain_graph
from network_compute import reciprocity

G,A,labels = brain_graph()

edges = G.edges()
edges = {edge:1 for edge in edges}

recip_counter = 0
for edge in edges:
    complementary_edge = (edge[1],edge[0])
    if edges.has_key(complementary_edge):
        recip_counter +=1

recip_counter /= 2 # double-counted above

recip_from_loop = recip_counter/len(edges)

recip_from_function = reciprocity(A)

print("Reciprocity from loop: %.3f" %(recip_from_loop))
print("Reciprocity from function: %.3f" %(recip_from_function))
