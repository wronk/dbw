"""
Created on Wed Nov 12 11:18:14 2014

@author: rkp, wronk

Functions for generating random binary undirected graphs not included in
NetworkX.
"""

import numpy as np
import networkx as nx
import graph_tools.auxiliary as aux_tools

from binary_directed import biophysical_reverse_outdegree


def ER_distance(N=426, p=.086, brain_size=[7., 7., 7.]):
    """Create an Erdos-Renyi random graph in which each node is assigned a
    position in space, so that relative positions are represented by a distance
    matrix."""
    # Make graph & get adjacency matrix
    G = nx.erdos_renyi_graph(N, p)
    A = nx.adjacency_matrix(G)
    # Randomly distribute nodes in space & compute distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A, D


def biophysical(N=426, N_edges=7804, L=2.2, gamma=1.75,
                brain_size=[7., 7., 7.]):
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
    G = nx.Graph()
    G.add_nodes_from(np.arange(N))

    # Randomly add edges
    edge_ctr = 0
    while edge_ctr < N_edges:
        # Update degree list & degree-related probability vector
        degs = A.sum(1).astype(float)
        degs_prob = degs.copy()

        # Pick random node to draw edge from
        from_idx = np.random.randint(low=0, high=N)

        # Skip this node if already fully connected
        if degs[from_idx] == N:
            continue

        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[from_idx, :] > 0
        degs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        degs_prob[from_idx] = 0

        # Calculate cxn probabilities from degree & distance
        P = (degs_prob ** gamma) * D_decay[from_idx, :]
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum())  # Normalize probabilities to sum to 1

        # Sample node from distribution
        to_idx = np.random.choice(np.arange(N), p=P)

        # Add edge to graph
        if A[from_idx, to_idx] == 0:
            G.add_edge(from_idx, to_idx, {'d': D[from_idx, to_idx]})

        # Add edge to adjacency matrix
        A[from_idx, to_idx] += 1
        A[to_idx, from_idx] += 1

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G, A, D


def undirected_biophysical_reverse_outdegree(N=426, N_directed_edges=8820,
                                             L=np.inf, gamma=1.75,
                                             brain_size=[7., 7., 7.]):
    """Identical to the biophysical reverse outdegree model, except that
    adjacency matrix is symmetrized so that reciprocal edges merge into one."""

    # create
    G, A, D = biophysical_reverse_outdegree(N=N, N_edges=N_directed_edges, L=L,
                                            gamma=gamma, brain_size=brain_size)

    # symmetrize graph
    G = G.to_undirected()

    return G, A, D


def random_simple_deg_seq(sequence, brain_size=[7., 7., 7.], seed=None,
                          tries=10):
    '''Wrapper function to get a SIMPLE (no parallel or self-loop edges) graph
    that has a given degree sequence.

    This graph is used conventionally as a control because it yields a random
    graph that accounts for degree distribution.

    Parameters:
        sequence: list of int
            Degree of each node to be added to the graph.
        brain_size: list of 3 floats
            Size of the brain to use when distributing  node locations.
        seed: hashable object for random seed
            Seed for the random number generator.
        tries: int
            Number of attempts at creating graph (function will retry if
            self-loops exist.
    Returns:
        Networkx graph object, adjacency matrix, and random distances'''

    G = nx.random_degree_sequence_graph(sequence=sequence, seed=seed,
                                        tries=tries)

    A = nx.adjacency_matrix(G)
    N = len(sequence)
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A, D
