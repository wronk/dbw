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

# Set parameters
p_th = .01 # P-value threshold
w_th = 0 # Weight-value threshold

# Set relative directory path
dir_name = '../Data/linear_model'

# Load weights & p-values
W,P,row_labels,col_labels = network_gen.load_weights(dir_name)
# Threshold weights according to weights & p-values
W_net,mask = network_gen.threshold(W,P,p_th=p_th,w_th=w_th)
# Set weights to zero if they don't satisfy threshold criteria
W_net[W_net==-1] = 0.
# Set diagonal weights to zero
np.fill_diagonal(W_net,0)

# Put everything in a dictionary
W_net_dict = {'row_labels':row_labels,'col_labels':col_labels,
              'data':W_net}

# Convert to networkx graph object
G = network_gen.import_weights_to_graph(W_net_dict)    

N = len(G.nodes())

# These are the new params, derived from adjusting the proverbial knobs
G_ER = nx.erdos_renyi_graph(N,0.085)
G_BA = nx.barabasi_albert_graph(N,19)
G_WS = nx.watts_strogatz_graph(N,36,0.159)

# Here you can specify which plotting function you want to run.
#x   It needs to take a single graph as input!
plotfunction = plot_net.plot_edge_btwn

myrange = np.linspace(0,0.002,50)

Fig,ax = plotfunction(G,myrange)
Fig_ER,ax_ER = plotfunction(G_ER,myrange)
Fig_BA,ax_BA = plotfunction(G_BA,myrange)
Fig_WS,ax_WS = plotfunction(G_WS,myrange)

if type(ax) == type(np.ndarray([0])):
    ax = ax[0]
    ax_ER = ax_ER[0]
    ax_BA = ax_BA[0]
    ax_WS = ax_WS[0]

xLims = [ax.get_xlim(), ax_ER.get_xlim(), ax_WS.get_xlim(), ax_BA.get_xlim()]
yLims = [ax.get_ylim(), ax_ER.get_ylim(), ax_WS.get_ylim(), ax_BA.get_ylim()]

xLims = [i for entry in xLims for i in entry]
yLims = [i for entry in yLims for i in entry]
MyLims = [min(xLims),max(xLims),min(yLims),max(yLims)]

ax.axis(MyLims)
ax_ER.axis(MyLims)
ax_WS.axis(MyLims)
ax_BA.axis(MyLims)

ax.set_title('Allen Mouse Brain Atlas (LM)')
ax_ER.set_title('Erdos-Renyi random network')
ax_BA.set_title('Barabasi-Albert scale-free network')
ax_WS.set_title('Watts-Strogatz small world network')

plt.show()