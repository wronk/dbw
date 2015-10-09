"""
Created on Wed Nov 12 11:18:14 2014

@author: rkp, wronk

Functions for generating random binary undirected graphs not included in
(or needing modification from) NetworkX.
"""

import numpy as np
import networkx as nx
import graph_tools.auxiliary as aux_tools
import aux_random_graphs

import brain_constants as bc


def ER_distance(N=bc.num_brain_nodes, p=bc.num_brain_edges_directed, brain_size=[7., 7., 7.]):
    """Create an directed Erdos-Renyi random graph in which each node is
    assigned a position in space, so that relative positions are represented
    by a distance matrix."""
    # Make graph & get adjacency matrix
    G = nx.erdos_renyi_graph(N, p, directed=True)
    A = nx.adjacency_matrix(G)
    # Randomly distribute nodes in space & compute distance matrix
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A.todense(), D


def biophysical(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed,
                L=np.inf, gamma=1.75, brain_size=[7., 7., 7.]):
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

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G, A, D


def biophysical_reverse(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed,
                        L=np.inf, gamma=1.75, brain_size=[7., 7., 7.]):
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
        unavail_mask = A[:, to_idx] > 0
        degs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        degs_prob[to_idx] = 0

        # Calculate cxn probabilities from degree & distance
        P = (degs_prob ** gamma) * D_decay[:, to_idx]
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum())  # Normalize probabilities to sum to 1

        # Sample node from distribution
        from_idx = np.random.choice(np.arange(N), p=P)

        # Add edge to graph
        if A[from_idx, to_idx] == 0:
            G.add_edge(from_idx, to_idx, {'d': D[from_idx, to_idx]})

        # Add edge to adjacency matrix
        A[from_idx, to_idx] += 1

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G, A, D


def biophysical_indegree(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed, L=np.inf,
                         gamma=1., brain_size=[7., 7., 7.]):
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
        indegs_prob = indegs.copy()

        # Pick random node to draw edge from
        from_idx = np.random.randint(low=0, high=N)

        # Skip this node if already fully connected
        if outdegs[from_idx] == N:
            continue

        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[from_idx, :] > 0
        indegs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        indegs_prob[from_idx] = 0

        # Calculate cxn probabilities from degree & distance
        P = (indegs_prob ** gamma) * D_decay[from_idx, :]
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

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G, A, D


def biophysical_reverse_outdegree(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed,
                                  L=np.inf, gamma=1., brain_size=(7., 7., 7.)):
    """Create a biophysically inspired graph. Source probability depends on
    outdegree. Target probability depends on distance.

    Args:
        N: how many nodes
        N_edges: how many edges
        L: length constant
        gamma: power to raise degree to
        brain_size: size of space in which nodes are randomly placed
    Returns:
        Networkx graph object, adjacency matrix, distance matrix"""

    # Pick node positions & calculate distance matrix
    if brain_size == 'brain':
        centroids_dict = aux_random_graphs.get_coords()
        labels = centroids_dict.keys()
        centroids = np.zeros((N, 3))

        for ctr, label in enumerate(labels):
            centroids[ctr, :] = centroids_dict[label] / 10.0

    else:
        centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))

    # Calculate distance matrix and distance decay matrix
    D = aux_tools.dist_mat(centroids)
    D_decay = np.exp(-D / L)

    # Initialize diagonal adjacency matrix
    A = np.eye(N, dtype=float)

    # Make graph object
    G = nx.DiGraph()
    G.add_nodes_from(np.arange(N))
    G.centroids = centroids

    # Randomly add edges
    edge_ctr = 0
    while edge_ctr < N_edges:
        # Update degree list & degree-related probability vector
        outdegs = A.sum(1).astype(float)
        outdegs_prob = outdegs.copy()

        # Calculate source node probability
        P = outdegs_prob ** gamma
        # On the off chance that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P.sum())  # Normalize probabilities to sum to 1

        # Sample node from distribution
        src_idx = np.random.choice(np.arange(N), p=P)

        D_src = D_decay[src_idx, :]

        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[src_idx, :] > 0
        D_src[unavail_mask] = 0

        # Set self-connection probability to 0
        D_src[src_idx] = 0

        D_src /= float(D_src.sum())

        # Pick random node to draw edge to
        targ_idx = np.random.choice(np.arange(N), p=D_src)

        # Skip this node if already fully connected
        if outdegs[src_idx] == N:
            continue

        # Add edge to graph
        if A[src_idx, targ_idx] == 0:
            G.add_edge(src_idx, targ_idx, {'d': D[src_idx, targ_idx]})

        # Add edge to adjacency matrix
        A[src_idx, targ_idx] += 1

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G, A, D


def biophysical_reverse_outdegree_nonspatial(N=bc.num_brain_nodes,
                                             N_edges=bc.num_brain_edges_directed, gamma=1.):
    """Create a biophysically inspired graph. Source probability depends on
    outdegree. Target probability is uniform at random.

    Args:
        N: how many nodes
        N_edges: how many edges
        gamma: power to raise outdegree to
    Returns:
        Networkx graph object"""

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
        outdegs_prob = outdegs.copy()

        # Pick random node to draw edge to
        to_idx = np.random.randint(low=0, high=N)

        # Skip this node if already fully connected
        if indegs[to_idx] == N:
            continue

        # Find unavailable cxns and set their probability to zero
        unavail_mask = A[:, to_idx] > 0
        outdegs_prob[unavail_mask] = 0
        # Set self cxn probability to zero
        outdegs_prob[to_idx] = 0

        # Calculate cxn probabilities from degree & distance
        P = outdegs_prob ** gamma
        # On the off changes that P == 0, skip
        if P.sum() == 0:
            continue
        # Otherwise keep going on
        P /= float(P. sum())  # Normalize probabilities to sum to 1

        # Sample node from distribution
        from_idx = np.random.choice(np.arange(N), p=P)

        # Add edge to graph
        if A[from_idx, to_idx] == 0:
            G.add_edge(from_idx, to_idx)

        # Add edge to adjacency matrix
        A[from_idx, to_idx] += 1

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G


def biophysical_reverse_outdegree_reciprocal(N=bc.num_brain_nodes, N_edges=bc.num_brain_edges_directed,
                                             gamma=1., reciprocity=10.):
    """Create a biophysically inspired graph. Source probability depends on
    outdegree. Target probability depends on whether reciprocal connections
    are present.

    Args:
        N: how many nodes
        N_edges: how many edges
        gamma: power to raise outdegree to
        reciprocity: probability ratio of connecting to already connected vs.
            unconnected targets
    Returns:
        Networkx graph object"""

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

        # Pick source node
        ## Set cxn prob of fully out-connected src nodes to zero
        outdegs_prob[outdegs == N] = 0

        ## Calculate src node probabilities from degree & distance
        psrc = outdegs_prob ** gamma
        ## Normalize psrc
        psrc /= float(psrc.sum())

        ## Sample source node from distribution
        from_idx = np.random.choice(np.arange(N), p=psrc)

        # Pick target node
        ## Initialize uniform probability array
        ptarg = np.ones((N,), dtype=float)
        ## Get mask of nodes that already have edges from source to target
        ptarg[A[from_idx, :] > 0] = 0
        ## Augment probability of connecting to nodes with edges from target to source
        ptarg[A[:, from_idx] > 0] *= reciprocity
        ## Normalize ptarg
        ptarg /= float(ptarg.sum())

        ## Sample target node from distribution
        to_idx = np.random.choice(np.arange(N), p=ptarg)

        # Add edge to graph
        if A[from_idx, to_idx] == 0:
            G.add_edge(from_idx, to_idx)

        # Add edge to adjacency matrix
        A[from_idx, to_idx] += 1

        # Increment edge counter
        edge_ctr += 1

    # Set diagonals to zero
    np.fill_diagonal(A, 0)

    return G


def random_directed_deg_seq(in_sequence, out_sequence, simplify,
                            brain_size=[7., 7., 7.], create_using=None,
                            seed=None):
    '''Wrapper function to get a MULTIGRAPH (parallel or self-loop edges may
    exist) given degree sequence. Chance of parallel/multiedges diminish
    as graph has more nodes.

    This graph is used conventionally as a control because it yields a random
    graph that accounts for degree distribution.

    Parameters:
    -----------
        in_sequence: list of int
            In degree of each node to be added to the graph.
        out_sequence: list of int
            Out degree of each node to be added to the graph.
        simplify: bool
            Whether or not to remove  self-loops and parallel edges. Will
            change degree sequence slightly, but effect diminishes with
            increasing size of graphs.
        brain_size: list of 3 floats
            Size of the brain to use when distributing  node locations.
        create_using: graph, optional
            Return graph of this type. The instance will be cleared.
        seed: hashable object for random seed
            Seed for the random number generator.
    Returns:
    --------
        Networkx graph object, adjacency matrix, and random distances'''

    # Create configuration model using specified properties
    G = nx.directed_configuration_model(in_sequence, out_sequence,
                                        create_using=create_using, seed=seed)
    if simplify:
        G = nx.DiGraph(G)  # Remove parallel edges
        G.remove_edges_from(G.selfloop_edges())  # Remove self loops

    # Get adjacency info and create random spatial locations
    A = nx.adjacency_matrix(G)
    N = len(in_sequence)
    centroids = np.random.uniform([0, 0, 0], brain_size, (N, 3))
    D = aux_tools.dist_mat(centroids)

    return G, A, D


def growing_SGPA_1(n_nodes, p_edge_split, l, brain_size, remove_extra_ccs):
    """
    Grow a source-growth-proximal-attachment graph by adding nodes one at a time.
    :param n_nodes: number of total nodes
    :param p_edge_split: probability that an edge splits in one time step
    :param l: length constant
    :param brain_size: size of brain to uniformly distribute node positions within
    :param prune: whether or not to remove nodes that have no edges
    :return: graph
    """

    def random_pos():
        return np.random.uniform([0, 0, 0], brain_size)

    # initialize new graph
    G = nx.DiGraph()
    G.add_node(0, {'pos': random_pos()})
    G.add_edge(0, 0)

    n_edges = 0

    # loop over edge splits and node additions
    for node_i in range(1, n_nodes):
        # add new node with self edge
        G.add_node(node_i, {'pos': random_pos()})
        G.add_edge(node_i, node_i)

        # add edges
        for edge in np.array(G.edges()).copy():
            if np.random.rand() < p_edge_split:
                src = edge[0]
                targs = [node for node in G.nodes() if node != src]
                # determine target
                dists = np.array(
                    [np.linalg.norm(G.node[src]['pos'] - G.node[targ]['pos']) for targ in targs]
                )
                probs = np.exp(-dists/l)
                probs /= probs.sum()
                targ = np.random.choice(targs, p=probs)
                G.add_edge(src, targ)
                n_edges += 1

    for node_i in range(n_nodes):
        G.remove_edge(node_i, node_i)

    if remove_extra_ccs:
        ccs = list(nx.connected_components(G.to_undirected()))
        max_cc_size = np.max([len(cc) for cc in ccs])
        for cc in ccs:
            if len(cc) != max_cc_size:
                [G.remove_node(node) for node in cc]

    print "{} edges".format(len(G.edges()))
    print "{} nodes".format(len(G.nodes()))
    return G