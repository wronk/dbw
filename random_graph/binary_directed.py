"""
Created on Wed Nov 12 11:18:14 2014

@author: rkp

Functions for generating random binary undirected graphs not included in 
NetworkX.
"""

import numpy as np
import networkx as nx
import graph_tools.auxiliary as aux_tools

def ER_distance(N=426, p=.043, brain_size=[7., 7., 7.]):
    """Create an directed Erdos-Renyi random graph in which each node is assigned a 
    position in space, so that relative positions are represented by a distance
    matrix."""
    # Make graph & get adjacency matrix
    G = nx.erdos_renyi_graph(N, p, directed=True)
    A = nx.adjacency_matrix(G)
    # Randomly distribute nodes in space & compute distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)
    
    return G, A, D

def biophysical(N=426, N_edges=8820, L=np.inf, gamma=1.7, brain_size=[7., 7, 7]):
    """Create a biophysically inspired graph. Connection probabilities depend
    on distance & degree.
    
    Args:
        N: how many nodes
        N_edges: how many edges
        L: length constant
        gamma: power to raise degree to
        brain_size: size of space in which nodes are randomly placed
    Returns:
        Networkx graph object, adjacency matrix, distance matrix"""
    # Pick node positions & calculate distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    
    # Calculate distance matrix and distance decay matrix
    D = aux_tools.dist_mat(centroids)
    D_decay = np.exp(-D / L)
    
    # Initialize diagonal adjacency matrix
    A = np.eye(N, dtype=float)
    
    # Make graph object
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))
    
    # Randomly add edges
    edge_ctr = 0
    while edge_ctr < N_edges:
        # Update degree list & degree-related probability vector
        indegs = A.sum(0).astype(float)
        outdegs = A.sum(1).astype(float)
        degs = indegs + outdegs
        degs_prob = degs.copy()
        
        # Pick random node to draw edge from
        from_idx = np.random.randint(low=0, high=N)
        
        # Skip this node if already fully connected
        if outdegs[from_idx] == N:
            continue
        
        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[from_idx,:] > 0
        degs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        degs_prob[from_idx] = 0
        
        # Calculate cxn probabilities from degree & distance
        P = (degs_prob**gamma) * D_decay[from_idx,:]
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum()) # Normalize probabilities to sum to 1
            
        # Sample node from distribution
        to_idx = np.random.choice(np.arange(N),p=P)
        
        # Add edge to graph
        if A[from_idx,to_idx] == 0:
            G.add_edge(from_idx,to_idx,{'d':D[from_idx,to_idx]})
            
        # Add edge to adjacency matrix
        A[from_idx,to_idx] += 1
        
        # Increment edge counter
        edge_ctr += 1
    
    # Set diagonals to zero
    np.fill_diagonal(A,0)
        
    return G, A, D

def biophysical_reverse(N=426, N_edges=8820, L=np.inf, gamma=1.7, brain_size=[7., 7, 7]):
    """Create a biophysically inspired graph. Connection probabilities depend
    on distance & degree.
    
    Args:
        N: how many nodes
        N_edges: how many edges
        L: length constant
        gamma: power to raise degree to
        brain_size: size of space in which nodes are randomly placed
    Returns:
        Networkx graph object, adjacency matrix, distance matrix"""
    # Pick node positions & calculate distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    
    # Calculate distance matrix and distance decay matrix
    D = aux_tools.dist_mat(centroids)
    D_decay = np.exp(-D / L)
    
    # Initialize diagonal adjacency matrix
    A = np.eye(N, dtype=float)
    
    # Make graph object
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))
    
    # Randomly add edges
    edge_ctr = 0
    while edge_ctr < N_edges:
        # Update degree list & degree-related probability vector
        indegs = A.sum(0).astype(float)
        outdegs = A.sum(1).astype(float)
        degs = indegs + outdegs
        degs_prob = degs.copy()
        
        # Pick random node to draw edge to
        to_idx = np.random.randint(low=0, high=N)
        
        # Skip this node if already fully connected
        if outdegs[to_idx] == N:
            continue
        
        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[:,to_idx] > 0
        degs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        degs_prob[to_idx] = 0
        
        # Calculate cxn probabilities from degree & distance
        P = (degs_prob**gamma) * D_decay[:,to_idx]
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum()) # Normalize probabilities to sum to 1
            
        # Sample node from distribution
        from_idx = np.random.choice(np.arange(N),p=P)
        
        # Add edge to graph
        if A[from_idx,to_idx] == 0:
            G.add_edge(from_idx,to_idx,{'d':D[from_idx,to_idx]})
            
        # Add edge to adjacency matrix
        A[from_idx,to_idx] += 1
        
        # Increment edge counter
        edge_ctr += 1
    
    # Set diagonals to zero
    np.fill_diagonal(A,0)
        
    return G, A, D
    
def biophysical_reverse_outdegree(N=426, N_edges=8820, L=np.inf, gamma=1.7, brain_size=[7., 7, 7]):
    """Create a biophysically inspired graph. Connection probabilities depend
    on distance & degree.
    
    Args:
        N: how many nodes
        N_edges: how many edges
        L: length constant
        gamma: power to raise degree to
        brain_size: size of space in which nodes are randomly placed
    Returns:
        Networkx graph object, adjacency matrix, distance matrix"""
    # Pick node positions & calculate distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    
    # Calculate distance matrix and distance decay matrix
    D = aux_tools.dist_mat(centroids)
    D_decay = np.exp(-D / L)
    
    # Initialize diagonal adjacency matrix
    A = np.eye(N, dtype=float)
    
    # Make graph object
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))
    
    # Randomly add edges
    edge_ctr = 0
    while edge_ctr < N_edges:
        # Update degree list & degree-related probability vector
        outdegs = A.sum(1).astype(float)
        outdegs_prob = outdegs.copy()
        
        # Pick random node to draw edge to
        to_idx = np.random.randint(low=0, high=N)
        
        # Skip this node if already fully connected
        if outdegs[to_idx] == N:
            continue
        
        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[:,to_idx] > 0
        outdegs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        outdegs_prob[to_idx] = 0
        
        # Calculate cxn probabilities from degree & distance
        P = (outdegs_prob**gamma) * D_decay[:,to_idx]
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum()) # Normalize probabilities to sum to 1
            
        # Sample node from distribution
        from_idx = np.random.choice(np.arange(N),p=P)
        
        # Add edge to graph
        if A[from_idx,to_idx] == 0:
            G.add_edge(from_idx,to_idx,{'d':D[from_idx,to_idx]})
            
        # Add edge to adjacency matrix
        A[from_idx,to_idx] += 1
        
        # Increment edge counter
        edge_ctr += 1
    
    # Set diagonals to zero
    np.fill_diagonal(A,0)
        
    return G, A, D