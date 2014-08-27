# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:10:06 2014

@author: sid
"""
import matplotlib.pyplot as plt
import numpy as np

def connection_strength(W,bins=10):
    # Returns figure/axis and plots a histogram of connection strength
    W_new = W[W>0]
    
    Fig,ax = plt.subplots(1,1)
    binnedW,bins,patches = plt.hist(W_new,bins, facecolor='red', alpha=0.5)
    
    
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Frequency')
    ax.set_title('Connection strength')
    plt.show()
    return Fig,ax
    
def shortest_path_distribution(W):
    # Returns figure/axis and plots a bar graph of shortest path distribution
    
    if type(W) != '<type \'numpy.ndarray\'>':
        W = np.array(W)
        
    W_new = W[W>0]
    uniques = np.unique(W_new)
    int_uniques = [int(entry) for entry in uniques]
    counts = []    
    
    for j in range(len(uniques)):
        current = uniques[j]
        counts.append(sum(W_new == current))
    
    Fig,ax = plt.subplots(1,1)
    
    ax.bar(uniques,counts)
    ax.set_xlabel('Number of nodes in shortest path')
    ax.set_ylabel('Frequency')
    ax.set_xticks(uniques+0.4)
    ax.set_xticklabels(int_uniques)
    ax.set_title('Distribution of shortest path lengths')
    
    plt.show()
    
    return Fig,ax