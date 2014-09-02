import pdb
"""
Created on Mon Sep  1 22:37:27 2014

@author: rkp

Code to generate a scale-free graph with adjustable clustering coefficient
according to the paper Herrera & Zufiria 2011.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def scale_free_cc_graph(n=426,m=12,p=np.array([0,1.]),
                        fp=np.array([.5,.5]),k0=None):
    """Generate a random graph with both scale-free properties and a tunable
    clustering coefficient distribution.
    
    The graph is initialized with n0 = max(10,m) nodes, as in the paper.
    
    Args:
        n: Number of nodes.
        m: Number of edges to add for each added node.
        p: Support of distribution over random walk length probabilities.
        fp: Probability of random walk length probabilities.
        k0: Starting average node degree.
    Returns:
        G: Networkx graph object with desired properties."""
    n0 = np.max([10,m])
    if k0 is None:
        k0 = m/2
    k0 = int(k0)
    
    # Create empty graph
    G = nx.empty_graph(n0)
    # Add walk length probabilities
    init_prob = np.random.choice(p,size=(n0,),p=fp)
    for node_idx in range(n0):
        G.node[node_idx]['p'] = init_prob[node_idx]
    
    # Add initial edges
    for node in range(n0):
        # Choose k0 connections
        available_nodes = np.array(range(0,node) + range(node+1,n0))
        selected_nodes = np.random.permutation(available_nodes)[:k0]
        # Add edges
        G.add_edges_from(zip([node]*k0,selected_nodes))
        
    # Grow graph
    for node in range(n0,n):
        # Mark down m nodes to connect to
        cxn_keys = [None for idx in range(m)]
        # Randomly pick first cxn according to degree distribution
        deg_tuples = zip(*G.degree().items())
        node_keys = deg_tuples[0]
        node_degs = np.array(deg_tuples[1])
        # Create probability distribution from degrees
        deg_prob = node_degs.astype(float)/node_degs.sum()
        cxn_keys[0] = np.random.choice(node_keys,p=deg_prob)
        # Random walk to get to other nodes
        cur_node = cxn_keys[0]
        cxn_ctr = 1
        while cxn_ctr < m:
            # Take one-step random walk
            next_node = np.random.choice(G.neighbors(cur_node))
            # Take second step with specified probability
            if np.random.rand() > G.node[cur_node]['p']:
                next_node = np.random.choice(G.neighbors(next_node))
            cur_node = next_node
            if cur_node not in cxn_keys:
                cxn_keys[cxn_ctr] = cur_node
                cxn_ctr += 1
        # Add new node with specific probability
        G.add_node(node,p=np.random.choice(p,p=fp))
        # Add edges
        G.add_edges_from(zip([node]*m,cxn_keys))
        
    return G
    


if __name__ == '__main__':
    G = scale_free_cc_graph(n=426,m=12,k0=3,p=np.array([1]),fp=np.array([1]))