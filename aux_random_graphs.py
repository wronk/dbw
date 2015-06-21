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
import scipy.io.matlab as scio

def dist_mat(centroids):
    """Compute a distance matrix from 3D centroids."""
    
    D = np.zeros((centroids.shape[0],centroids.shape[0]),dtype=float)
    for r_idx in range(D.shape[0]):
        for c_idx in range(D.shape[1]):
            d = np.sqrt(np.sum((centroids[r_idx,:] - centroids[c_idx])**2))
            D[r_idx,c_idx] = d
    return D
    
    
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
        # Choose k connections
        k = np.random.random_integers(1,k0)
        available_nodes = np.array(range(0,node) + range(node+1,n0))
        selected_nodes = np.random.permutation(available_nodes)[:k]
        # Add edges
        G.add_edges_from(zip([node]*k,selected_nodes))
        
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
    

def biophysical_graph(N=426,N_edges=7804,L=1.,power=1.5,dims=[10.,10,10],mode=0,centroids=0):
    """Create a biophysically inspired graph. Connection probabilities depend
    on distance & degree.
    
    Returns:
        Networkx graph object."""
    # Pick node positions & calculate distance matrix
    # If we haven't specified centroids, just use some randomly chosen one.
    
    G = nx.Graph()
    if type(centroids) == dict:
	centroids_dict = centroids.copy()
	keys = centroids.keys()
	G.add_nodes_from(keys)
	centroids = np.zeros([N,3])
	# This is so that we can have the centroid coords in the graph structure
	for k in range(N):
	    centroids[k,:] = centroids_dict[keys[k]]
	    G.node[keys[k]] = centroids_dict[keys[k]]
    else:
	centroids = np.random.uniform([0,0,0],dims,(N,3))
	keys = np.arange(N)
	G.add_nodes_from(keys)
	for k in range(N):
	    G.node[k] = centroids[k,:]
	
	    
	
    
    
    # Calculate distance matrix
    D = dist_mat(centroids)
    #D_exp = np.exp(-D**2/(2*L**2)) # This is currently Gaussian
    D_exp = np.exp(-D/L)
    np.fill_diagonal(D_exp,0)
    # Initialize diagonal adjacency matrix
    A = np.eye(N,dtype=float)
    # Make graph object
    
    
    
    # Randomly add edges
    for edge in range(N_edges):
        # Update degree list
        degs = A.sum(1)
        # Pick random node to draw edge from
        node0_idx = np.random.choice(np.arange(N))
        if mode == 0:
            # Continue if this node is already fully connected
            if degs[node0_idx] == N-1:
                print 'whoops'
                continue
            # Make list of impossible connections
            unavail_mask = A[node0_idx,:] > 0
            # Set unconnectable node degrees to zero to get zero cxn prob
            degs[unavail_mask] = 0
            # Calculate unnormalized connection probabilities
            P_un = (degs**power)*D_exp[node0_idx,:]
        elif mode == 1:
            P_un = degs*D_exp[node0_idx,:]
        # Normalize probabilities
        P = P_un/float(P_un.sum())
            
#        ax.cla()
#        ax.bar(np.arange(len(P)),P)
#        plt.draw()
        # Sample node from distribution
        node1_idx = np.random.choice(np.arange(N),p=P)
        # Add edge to graph
        if A[node0_idx,node1_idx] == 0:
            G.add_edge(keys[node0_idx],keys[node1_idx],{'d':D[node0_idx,node1_idx]})
        # Add edge to adjacency matrix
        A[node0_idx,node1_idx] += 1
        A[node1_idx,node0_idx] += 1
       
    return G,A,D

def get_coords():
    import os
    
    data_dir = os.getenv('DBW_DATA_DIRECTORY')
    A = scio.loadmat(data_dir+'/Structure_coordinates_and_distances.mat')
    contra_coords = A['cntrltrl_coordnt_dctnry']
    contra_type = contra_coords.dtype
    contra_names = contra_type.names
    contra_dict = {contra_names[k] + '_L':contra_coords[0][0][k][0] for k in range(len(contra_coords[0][0]))}

    ipsi_coords = A['ipsltrl_coordnt_dctnry']
    ipsi_type = ipsi_coords.dtype
    ipsi_names = ipsi_type.names
    ipsi_dict = {ipsi_names[k] + '_R':ipsi_coords[0][0][k][0] for k in range(len(ipsi_coords[0][0]))}

    contra_dict.update(ipsi_dict)
    
    return contra_dict
    
    
def get_distances():
    # Might want to compute your own Euclidean distances here?
    A = get_coords()
    names = A.keys()
    D = np.zeros([len(A),len(A)])
    for i in range(len(A)):
	current_i = names[i]
	for j in range(len(A)):
	    current_j = names[j]
	   
	    D[i,j] = sum((A[current_i] - A[current_j])**2)
	   
    return D,names
    
def get_edge_distances(G):
    edges = G.edges()
    distances = {}
    for edge in edges:
	centroid1 = G.node[edge[0]]
	centroid2 = G.node[edge[1]]
	d = sum((centroid1-centroid2)**2)
	distances[edge] = d
	
    return distances
    
def compute_average_distance(G):
    distances = get_edge_distances(G)
    avg_distance = {}
    for node in G.nodes():
	current_distances = [distances[edge] for edge in distances if node in edge]
	avg_distance[node] = np.mean(current_distances)
	
    return avg_distance
    
if __name__ == '__main__':
    G = scale_free_cc_graph(n=426,m=12,k0=3,p=np.array([1]),fp=np.array([1]))
