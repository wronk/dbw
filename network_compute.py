import pdb
"""
Created on Wed Aug 27 22:31:09 2014

@author: rkp

Set of functions for pulling out the nodes and edges according to specific
ranking criteria.
"""

import numpy as np
import operator


def reciprocity(W_net):
    """Calculate the percentage of reciprocal connections."""
    W_binary = W_net > 0
    np.fill_diagonal(W_binary,False)
    total_cxns = W_net.sum()
    recip_cxns = (W_net*W_net.T).sum()
    arecip_cxns = total_cxns - recip_cxns
    
    recip_coeff = recip_cxns/(recip_cxns + 2*arecip_cxns)
    return recip_coeff
    
def out_in(W_net, labels,binarized=True):
    """Calculate the output/input ratio given the weight matrix."""
    if binarized:
        W = (W_net > 0).astype(float)
    else:
        W = W_net.copy()
    # Calculate total output & input connections
    out_total = W.sum(axis=1)
    in_total = W.sum(axis=0)
    out_in_vec = out_total.astype(float) / in_total
    # Put into dictionary format
    out_dict = {labels[idx]: out_total[idx] for idx in range(len(labels))}
    in_dict = {labels[idx]: in_total[idx] for idx in range(len(labels))}
    out_in_dict = {labels[idx]: out_in_vec[idx] for idx in range(len(labels))}

    return out_dict, in_dict, out_in_dict


def get_ranked(criteria_dict, high_to_low=True):
    """Get labels & criteria, sorted (ranked) by criteria."""

    dict_list_sorted = sorted(criteria_dict.iteritems(),
                              key=operator.itemgetter(1), reverse=high_to_low)

    labels_sorted = [item[0] for item in dict_list_sorted]
    criteria_sorted = [item[1] for item in dict_list_sorted]

    return labels_sorted, criteria_sorted
    
    
def node_edge_overlap(node_list,edge_list):
    """Calculate the overlap of a set of nodes and edges.
    Returns which edges are touching a node and which connect two nodes."""
    
    # Calculate how many edges contain at least one node in node list
    edges_touching = [edge for edge in edge_list if edge[0] in node_list
                      or edge[1] in node_list]
    edges_connecting = [edge for edge in edge_list if edge[0] in node_list
                        and edge[1] in node_list]
                        
    return edges_touching, edges_connecting
    
def bidirectional_metrics(W_net, coords, labels, binarized=False):
    """Calculate bidirectionality metrics for a graph given its weights.
    
    Returns:
        List of labeled nonzero edges, (Ne x 3) array of distance, 
        bidirectionality coefficient, and connection strength."""
    if binarized:
        W_bi = (W_net > 0).astype(float)
    else:
        W_bi = W_net.copy()
    
    # Get nonzero elements of W_bi
    nz_idxs = np.array(W_bi.nonzero()).T
    nz_idxs = np.array([nz_idx for nz_idx in nz_idxs
                        if labels[nz_idx[0]][:-2] != labels[nz_idx[1]][:-2]])
    
    # Generate edge labels
    edges = [(labels[nz_idx[0]],labels[nz_idx[1]]) for nz_idx in nz_idxs]
    
    # Make array for storing bidirectional metrics
    bd_metrics = np.zeros((len(edges),3),dtype=float)
    
    # Calculate all metrics
    for e_idx,nz in enumerate(nz_idxs):
        # Distance
        d = np.sqrt(np.sum((coords[nz[0],:] - coords[nz[1],:])**2))
        # Strength
        s = W_bi[nz[0],nz[1]] + W_bi[nz[1],nz[0]]
        # Bidirectionality coefficient
        bdc = 1 - np.abs(W_bi[nz[0],nz[1]] - W_bi[nz[1],nz[0]])/s
        # Store metrics
        bd_metrics[e_idx,:] = [d,bdc,s]
        
    return edges, bd_metrics