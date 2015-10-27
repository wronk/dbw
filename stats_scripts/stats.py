import pandas as pd
import numpy as np
import networkx as nx
import os
import scipy.stats as stats

from extract.brain_graph import binary_directed
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model

from brain_constants import *
from network_compute import reciprocity


def get_gamma(G):
    # Fits clustering coefficient as a power of degree.
    # Usage: gamma,R2,p = get_gamma(G)
    # Input:
    # G: an undirected networkx graph object.
    # Returns:
    # gamma: the fitted power
    # R2: the R^2 value (percent of variance accounted for by power law)
    # p: the associated p-value.
    
    nodes = G.nodes()
    deg_dict = G.degree()
    cc_dict = nx.clustering(G)
    deg = [deg_dict[node] for node in nodes]
    cc = [cc_dict[node] for node in nodes]
    
    vals = stats.linregress(np.log(deg),np.log(cc))
    gamma = vals.slope; R2 = vals.rvalue**2; p = vals.pvalue
    return gamma,R2,p

def in_out_corr(G,connectome=False):
    # Computes Spearman correlation between in- and out-degree.
    # Usage: r,p = in_out_corr(G)
    # Input:
    # G: a directed networkx graph object
    # Returns:
    # r: Spearman rank correlation coefficient
    # p: the associated p-value
    #
    # Note: The argument connectome=True should be set if running
    # with the connectome. The connectome has 213 independent
    # nodes because it is mirrored along the hemispheric divide
    # in order to construct a complete graph.
    
    nodes = G.nodes()
    indeg_dict = G.in_degree()
    outdeg_dict = G.out_degree()
    indeg = [indeg_dict[node] for node in nodes]
    outdeg = [outdeg_dict[node] for node in nodes]

    # This is because the connectome only has n_nodes/2 independent
    # samples so we only use each sample once
    if connectome:
        indeg = np.array(indeg)[0:len(nodes)/2]
        outdeg = np.array(outdeg)[0:len(nodes)/2]

    corr = stats.spearmanr(indeg,outdeg)
    r = corr[0]
    p = corr[1]

    return r,p

def cc_deg_corr(G,connectome=False):
    # Correlation between clustering and degree.
    # Usage: r,p = cc_deg_corr(G)
    # Input:
    # G: an undirected networkx graph object
    # Returns:
    # r: Spearman rank correlation coefficient
    # p: the associated p-value
    #
    # Note: The argument connectome=True should be set if running
    # with the connectome. The connectome has 213 independent
    # nodes because it is mirrored along the hemispheric divide
    # in order to construct a complete graph.
    
    nodes = G.nodes()
    cc_dict = nx.clustering(G)
    deg_dict = G.degree()

    cc = [cc_dict[node] for node in nodes]
    deg = [deg_dict[node] for node in nodes]
    if connectome:
        cc = np.array(cc)[0:len(nodes)/2]
        deg = np.array(deg)[0:len(nodes)/2]

    corr = stats.spearmanr(cc,deg)
    r = corr.correlation; p = corr.pvalue
    return r,p

def get_edge_distances(G,split_recip=False):
    # Compute edge distances for graph. Graph must have a custom-set field
    # called centroids which specifies the XYZ coordinates of the node.
    # Usage; edges = get_edge_distances(G)
    # Returns:
    # edges: An array containing all the edge lengths of the network
    # OR
    # Usage: nonrecip_edges,recip_edges = get_edge_distances(G,split_recip=True)
    # nonrecip_edges: Edge lengths of all the nonreciprocal edges in the network
    # recip_edges: Edge lengths of all the reciprocal edges in the network
    #
    # Note: The optional argument split_recip=True will make the function return two
    # arrays: one for nonreciprocal and one for reciprocal edges.
    
    centroids = G.centroids

    indeg = np.array([G.in_degree()[node] for node in G])
    outdeg = np.array([G.out_degree()[node] for node in G])

    edges = {} # distances
    actual_edges = G.edges()

    # List of all edges and the the Euclidean distance
    recip_names = []
    nonrecip_names = []
    for edge in actual_edges:
        edges[edge] =  np.sqrt(np.sum((centroids[edge[0]] - centroids[edge[1]])**2))
        if edges.has_key((edge[1],edge[0])):
            if (edge[1],edge[0]) not in recip_names:
                recip_names.append(edge)
        else:            
            nonrecip_names.append(edge)
        
    nonrecip_edges = [edges[k] for k in nonrecip_names]
    recip_edges = [edges[k] for k in recip_names]


    if split_recip:
        return nonrecip_edges,recip_edges
    else:
        return edges

def cohen_d(x,y):
    # Computes Cohen's d (similar to d')
    # Usage: d = cohen_d(x,y)
    # Input:
    # x and y: two normally distributed vectors
    # Returns:
    # d: Cohen's d (difference between mean(x) and mean(y), expressed
    # in units of the pooled SD)
    
    pooled_var = (np.var(x)*(len(x)-1) + np.var(y)*(len(y)-1))/(len(x)+len(y)-2)
    diff = np.abs(np.mean(x)-np.mean(y))
    return diff/np.sqrt(pooled_var)


def distance_cc_corr(G):
    # Rank correlation between distance (edge lengths) and clustering coefficient.
    # Usage: rho,p = distance_cc_corr(G)
    # Input:
    # G: An undirected networkx graph object
    # Returns:
    # rho: rank correlation coefficient
    # p: the associated p-value
    
    nodes = G.nodes()
    edge_distances = get_edge_distances(G)
    mean_edge_distances = {}
    for node in G.nodes():        
        current_edges = []
        for edge in edge_distances:
            if node in edge:
                current_edges.append(edge_distances[edge])
        mean_edge_distances[node] = np.mean(current_edges)
        
    cc_dict = nx.clustering(G.to_undirected()); cc = [cc_dict[node] for node in nodes]
    edists = [mean_edge_distances[node] for node in nodes]
    distance_clustering = stats.spearmanr(edists,cc)
    
    rho = distance_clustering.correlation
    p =  distance_clustering.pvalue
    
    return rho,p


if __name__ == "__main__":
    run_graphs = True
    n_runs = 10

    if n_runs%2 == 0: # this must be odd, otherwise the median will be an average
        n_runs += 1

    
    G, _, labels = binary_directed()
    nodes = G.nodes()
    new_labels = {nodes[i]:labels[i] for i in range(len(nodes))}
    centroids = G.centroids
    G = nx.relabel_nodes(G,new_labels)
    G.centroids = centroids

    # Page 2: In the connectome, clustering coefficient can be expressed as a power of degree:
    # C ~ k^gamma, where gamma = -0.44
    # Also give corresponding values for SG and SGPA networks
    
    print '=== Clustering coefficients as a power of degree (C ~ k^gamma) ==='
    gamma,R2,p = get_gamma(G.to_undirected())
    print 'Mouse connectome: C ~ k^%.2f, R^2=%.3f, p=%.3e'%(gamma,R2,p)
    

    # Power laws for SG and SGPA; page 9
    if run_graphs:
        cc_deg_stats = {'SG':np.zeros((n_runs,3)),'SGPA':np.zeros((n_runs,3))}
        deg_stats = {'SG':np.zeros((n_runs,2)),'SGPA':np.zeros((n_runs,2))}        
        graphs = {'SG':np.inf,'SGPA':0.75}
        dist_cc_stats = {'SG':np.zeros((n_runs,2)),'SGPA':np.zeros((n_runs,2))}
        

    for g in graphs:
        if run_graphs:
            for k in range(n_runs):
                G_mod, A_mod, _ = biophysical_model(N=num_brain_nodes,
                                                    N_edges=num_brain_edges_directed,
                                                    L=graphs[g], gamma=1., brain_size=[7.,7.,7.])
        
                # This checks if clustering coefficient can be expressed as a power of degree
                cc_deg_stats[g][k,:] = get_gamma(G_mod.to_undirected())
                deg_stats[g][k,:] = in_out_corr(G_mod)
                dist_cc_stats[g][k,:] = distance_cc_corr(G_mod)

                # Get in and out-degree distributions

        gammas = cc_deg_stats[g][:,0].tolist()
        median_gamma = np.median(gammas)
        median_index = gammas.index(median_gamma)    
        R2 = cc_deg_stats[g][median_index,1]
        p = cc_deg_stats[g][median_index,2]
        print '%s model: C ~ k^%.2f, R^2=%.3f, p=%.3e'%(g,median_gamma,R2,p)    

        

    print ''
    print '=== Correlation between in- and out-degree ==='
    # Correlation between in- and out-deg for mouse connectome; page 6
    r_deg,p_deg = in_out_corr(G,connectome=True)
    print 'Mouse connectome: r = %.3f, p = %.3e'%(r_deg,p_deg)

    for g in graphs:
        r_deg = deg_stats[g][:,0].tolist()
        r_deg_median = np.median(r_deg)
        deg_median_index = r_deg.index(r_deg_median)
        p_deg_median = deg_stats[g][deg_median_index,1]
        print '%s model: r = %.3f, p = %.3e'%(g,r_deg_median,p_deg_median)
        
    print ''

    # Reciprocal edges are on average shorter than nonreciprocal edges (page 10)
    print '=== Reciprocal and nonreciprocal edge distances ==='
    
    nonrecip_distances,recip_distances = get_edge_distances(G,split_recip=True)
    t = stats.ttest_ind(nonrecip_distances,recip_distances)
    df = len(nonrecip_distances)+len(recip_distances)-2
    d = cohen_d(nonrecip_distances,recip_distances)
    print 'Connectome: t(%i)=%.2f, p=%.2e, Cohen\'s d=%.2f'%(df,t.statistic,t.pvalue,d)


    print ''
    print '=== Spearman correlation between edge distance and clustering coefficient ==='
    # Spearman correlation between edge distance and clustering, showing spatially localized topological clustering
    # Page 10
    # For the model:
    rho,p = distance_cc_corr(G)

    print 'Connectome: rho=%.3f, p=%.2e'%(rho,p)
    for g in graphs:
        r_dist_cc = dist_cc_stats[g][:,0].tolist()
        r_dist_cc_median = np.median(r_dist_cc)
        rdcm_index = r_dist_cc.index(r_dist_cc_median)
        p_dist_cc_median = dist_cc_stats[g][rdcm_index,1]
        rho_temp = r_dist_cc_median; p_temp = p_dist_cc_median
        print '%s model: rho=%.3f, p=%.2e'%(g,rho_temp,p_temp)
