"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for binary undirected graphs.
"""

import numpy as np
import networkx as nx


def reciprocity(G):
    """Calculate the reciprocity coefficient of a directed graph.
    
    Reciprocity is defined as (Ndirected - Nundirected)/Nundirected"""
    
    # get number of directed edges
    Ndirected = len(G.edges())
    
    # get number of undirected edges
    Nundirected = len(G.to_undirected().edges())
    
    # return reciprocity
    return (Ndirected - Nundirected) / float(Nundirected)