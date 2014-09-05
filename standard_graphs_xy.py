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
import aux_random_graphs

# Set parameters
p_th = .01 # P-value threshold
w_th = 0 # Weight-value threshold

# Set relative directory path
#dir_name = '../friday-harbor/linear_model'
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
G_BA_cc = nx.powerlaw_cluster_graph(N,19,1)

# Here you can specify which plotting function you want to run.
#x   It needs to take a single graph as input!
plotfunction = plot_net.plot_edge_btwn

myrange = np.linspace(0,0.002,20)

MyX = nx.betweenness_centrality

x_ER = MyX(G_ER).values()
y_ER = nx.clustering(G_ER).values()
#y_ER = G_ER.degree().values()

x_BA = MyX(G_BA).values()
#y_BA = G_BA.degree().values()
y_BA = nx.clustering(G_BA).values()

x_WS = MyX(G_WS).values()
#y_WS = G_WS.degree().values()
y_WS = nx.clustering(G_WS).values()


x_BA_cc = MyX(G_BA_cc).values()
#y_BA_cc = G_BA_cc.degree().values()
y_BA_cc = nx.clustering(G_BA_cc).values()

x = MyX(G).values()
#y = G.degree().values()
y = nx.clustering(G).values()

Fig,axs = plt.subplots(nrows=2,ncols=2,sharex=False, sharey=False, facecolor='White')

axs[0,0].scatter(x,y)

axs[0,1].scatter(x_ER,y_ER)

axs[1,0].scatter(x_BA,y_BA)

axs[1,1].scatter(x_BA_cc,y_BA_cc)

XTicks = [0,0.25,0.5,0.75,1]
XTicks = [0,0.0125,0.025,0.0375,0.05]
YTicks = [0,50,100,150,200]

for i in [0,1]:
    for j in [0,1]:
        #axs[i,j].set_xlim(-.01,0.05)
        #axs[i,j].set_ylim(0,200)
        axs[i,j].tick_params(labelsize=20)
        #axs[i,j].set_xticks(XTicks)
        #axs[i,j].set_yticks(YTicks)
# For clustering
        


TitleFontSize = 22
LabelFontSize = 20
axs[0,0].set_title('Allen Mouse Brain Atlas (LM)', fontsize=TitleFontSize)
axs[0,1].set_title('Watts-Strogatz small world network', fontsize=TitleFontSize)
axs[1,0].set_title('Symmetric Barabasi-Albert scale-free network', fontsize=TitleFontSize)
axs[1,1].set_title('Clustered scale-free network', fontsize=TitleFontSize)

ax_BA.set_xlabel('Clustering coefficient', fontsize=LabelFontSize)
ax_WS.set_xlabel('Clustering coefficient', fontsize=LabelFontSize)
ax.set_ylabel('Degree', fontsize=LabelFontSize)
ax_BA.set_ylabel('Degree', fontsize=LabelFontSize)

plt.draw()