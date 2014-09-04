# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 23:16:17 2014

@author: sid
"""

import networkx as nx
import plot_net
import matplotlib.pyplot as plt
import numpy as np
import network_gen
from networkx.generators.classic import empty_graph


def random_subset(S,m):
    L = []
    L_index = []
    assert len(np.unique(S)) >= m, "Uniques in S must be greater than m."
    
    while len(L) < m:
        index = np.random.randint(0,len(S))
        element = S[index]
        if element not in L:
            L.append(element)
            L_index.append(index)
    return L,L_index



def symmetric_BA_graph(n,m,p):
    """
    Return topologically symmetric Barabasi-Albert graph.
    n: number of nodes
    m: number of edges per node
    p: probability of contralateral edge
    
    Loosely based on the Barabasi-Albert graph in networkx
    """
    
    #my_p = 1/10

    if m < 1 or m >=n:
        raise nx.NetworkXError(\
        "Barabási-Albert network must have m>=1 and m<n, m=%d,n=%d"%(m,n))
        
    assert(n//2 > m), "n must be more than twice the size of m for symmetric graph"
    
    # Initialise the graph
    G=empty_graph(m)
    G.name="barabasi_albert_graph(%s,%s)"%(n,m)
    # Target nodes for new edges; both ipsilateral and contraletaral
    IpsiTargets=list(range(m))
    ContraTargets=list(range(n//2,n//2+m))
    
    # List of existing nodes, with nodes repeated once for each adjacent edge
    # This is a list of degrees 
    IpsiRepeatedNodes=[]
    ContraRepeatedNodes=[]
    # Start adding the other n-m nodes. The first node is m.
    IpsiSource = m
    ContraSource = m+ n//2
    
    # Set of nodes for ipsilateral and contralateral side
    IpsiNodes = set(range(n//2))
    ContraNodes = set(range(n//2,n))
    
    # List for matching ipsilateral to contralateral nodes
    IpsiNodesList = list(IpsiNodes)
    ContraNodesList = list(ContraNodes)
    
    
    
    First = 1
    while IpsiSource<n//2:
        # First we initialise the graph
        if First: # If first iteration
            First = 0
            IpsiInitEdges = []
            ContraInitEdges = []
            for k in range(len(IpsiTargets)): # Iterate through all targets
                # This section makes it so that there is a 33% chance to
                # assign the mth node to the ipsi, contra 
                p_connected = np.random.random(1)[0]
                if p_connected < 1./3.:
                    IpsiInitEdges.extend([IpsiTargets[k]])
                    ContraInitEdges.extend([ContraTargets[k]])
                    
                    # Then connect ipsi
                elif p_connected < 2./3.:
                    IpsiInitEdges.extend([ContraTargets[k]])
                    ContraInitEdges.extend([IpsiTargets[k]])
                    # Then connect contra
                else:
                    IpsiInitEdges.extend([IpsiTargets[k],ContraTargets[k]])
                    ContraInitEdges.extend([IpsiTargets[k],ContraTargets[k]])
                    
            # Add edges to the graph according to the pseudo-random rule above
            G.add_edges_from(zip([IpsiSource]*len(IpsiInitEdges), IpsiInitEdges))
            G.add_edges_from(zip([ContraSource]*len(ContraInitEdges), ContraInitEdges))
            
            # All nodes in the same list
            AllNodes = IpsiInitEdges + ContraInitEdges
            
            IpsiNodesInit = []
            ContraNodesInit = []
            
            # Work backwards to figure out how many degrees we've added to
            # the different ipsi/contralateral nodes.
            for k in range(len(AllNodes)):
                element = AllNodes[k]
                if element in IpsiNodes:
                    IpsiNodesInit.append(element)
                    # ContraNodes is found by taking the index of 
                    # the Ipsi node. This is okay because number of
                    # contra nodes should be the same as number of
                    # ipsi nodes (otherwise it's not symmetric!)
                    ContraNodesInit.append(ContraNodesList[element])
            
            # Add the nodes to the list of nodes (degree list, basically)
            # For ipsi side
            IpsiRepeatedNodes.extend(IpsiNodesInit)
            IpsiRepeatedNodes.extend([IpsiSource]*len(IpsiInitEdges))
            # For lateral side
            ContraRepeatedNodes.extend(ContraNodesInit)
            ContraRepeatedNodes.extend([ContraSource]*len(ContraInitEdges))
            
            # Iterate counter
            IpsiSource +=1
            ContraSource+=1
        else:
            # Number of ipsi/contralateral projections
            # Determined (pseudo-) probabilistically
            n_Ipsi = sum(np.random.random(m) > p)
            n_Contra = m-n_Ipsi
            
            # Take random subset of nodes from the degree list.
            IpsiTargetSet,Indices = random_subset(IpsiRepeatedNodes,m)
            # Match these up with the corresponding contralateral nodes
            # to get the appropriate symmetry.
            ContraTargetSet = [ContraRepeatedNodes[k] for k in Indices]
            
            # The if statements here is so that you always return a list.
            # Get a random sequence of indices...
            if n_Ipsi == 1:
                IpsiIndices = [np.random.permutation(range(m))[range(n_Ipsi)]]
            elif n_Ipsi:
                IpsiIndices = np.random.permutation(range(m))[range(n_Ipsi)]
            else:
                IpsiIndices = []
                
            if n_Contra == 1:
                ContraIndices = [np.random.permutation(range(m))[range(n_Contra)]]
            elif n_Contra:
                ContraIndices = np.random.permutation(range(m))[range(n_Contra)]
            else:
                ContraIndices = []
            
            # Apply the indices to the target set (and invert for contralateral so that we get symmetric projections)
            IpsiTargets = [IpsiTargetSet[k] for k in IpsiIndices] + [ContraTargetSet[k] for k in ContraIndices]
            ContraTargets = [ContraTargetSet[k] for k in IpsiIndices] + [IpsiTargetSet[k] for k in ContraIndices]
            
            
            # Add edges to the appropriate nodes
            G.add_edges_from(zip([IpsiSource]*len(IpsiTargets),IpsiTargets))
            G.add_edges_from(zip([ContraSource]*len(ContraTargets),ContraTargets))
            
            AllTargets = IpsiTargets + ContraTargets
            IpsiToAdd = []
            ContraToAdd = []
            
            # Again we work backwards to figure out which one we added
            for k in range(len(AllTargets)):
                element = AllTargets[k]
                if element in IpsiNodes:
                    IpsiToAdd.append(element)
                    ContraToAdd.append(ContraNodesList[element])
                    
            # Add nodes to list of degrees for ipsi and contralateral
            IpsiRepeatedNodes.extend(IpsiToAdd)
            IpsiRepeatedNodes.extend([IpsiSource]*len(IpsiTargets))
            
            ContraRepeatedNodes.extend(ContraToAdd)
            ContraRepeatedNodes.extend([ContraSource]*len(ContraTargets))
            
            IpsiSource += 1
            ContraSource += 1

    return G

#G = symmetric_BA_graph(426,19,0.52)
#G = symmetric_BA_graph(12,5,0.5)
#nx.draw(G)
#=======
