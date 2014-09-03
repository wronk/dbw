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
dir_name = '../friday-harbor/linear_model'

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
G_BA_cc = aux_random_graphs.scale_free_cc_graph(n=N,m=25,k0=20,p=np.array([1]),fp=np.array([1]))

# Here you can specify which plotting function you want to run.
#x   It needs to take a single graph as input!
plotfunction = plot_net.plot_edge_btwn

myrange = np.linspace(0,0.002,20)

MyX = nx.clustering

x_ER = MyX(G_ER).values()
y_ER = G_ER.degree().values()

x_BA = MyX(G_BA).values()
y_BA = G_BA.degree().values()

x_WS = MyX(G_WS).values()
y_WS = G_WS.degree().values()

x_BA_cc = MyX(G_BA_cc).values()
y_BA_cc = G_BA_cc.degree().values()

x = MyX(G).values()
y = G.degree().values()

Fig,axs = plt.subplots(nrows=3,ncols=2,sharex=False, sharey=False, facecolor='White')
ax = axs[0,0]; ax_ER = axs[0,1]; ax_BA = axs[1,0]; ax_WS = axs[1,1]; ax_BA_cc = axs[2,0]

ax.scatter(x,y)

ax_ER.scatter(x_ER,y_ER)

ax_BA.scatter(x_BA,y_BA)

ax_WS.scatter(x_WS,y_WS)

ax_BA_cc.scatter(x_BA_cc,y_BA_cc)

#Fig,ax = plotfunction(G,myrange)
#Fig_ER,ax_ER = plotfunction(G_ER,myrange)
#Fig_BA,ax_BA = plotfunction(G_BA,myrange)
#Fig_WS,ax_WS = plotfunction(G_WS,myrange)



#xLims = [ax.get_xlim(), ax_ER.get_xlim(), ax_WS.get_xlim(), ax_BA.get_xlim()]
#yLims = [ax.get_ylim(), ax_ER.get_ylim(), ax_WS.get_ylim(), ax_BA.get_ylim()]
#
#xLims = [i for entry in xLims for i in entry]
#yLims = [i for entry in yLims for i in entry]
#MyLims = [0,max(xLims),0,max(yLims)]
#MyLims = [0,1,0,max(yLims)]
#
#ax.axis(MyLims)
#ax_ER.axis(MyLims)
#ax_WS.axis(MyLims)
#ax_BA.axis(MyLims)

for a in [ax,ax_ER,ax_BA,ax_WS,ax_BA_cc]:
    a.set_xlim(0,1)
    a.set_ylim(0,200)
# For clustering
XTicks = [0,0.25,0.5,0.75,1]
YTicks = [0,50,100,150,200]

ax_BA.set_xticks(XTicks)
ax_WS.set_xticks(XTicks)

ax.set_yticks(YTicks)
ax_BA.set_yticks(YTicks)


ax_BA.tick_params(size=10,labelsize=16)
ax_WS.tick_params(size=10,labelsize=16)
ax.tick_params(size=10,labelsize=16)
ax_ER.tick_params(size=10,labelsize=16)

TitleFontSize = 22
LabelFontSize = 20
ax.set_title('Allen Mouse Brain Atlas (LM)', fontsize=TitleFontSize)
ax_ER.set_title('Erdos-Renyi random network', fontsize=TitleFontSize)
ax_BA.set_title('Barabasi-Albert scale-free network', fontsize=TitleFontSize)
ax_WS.set_title('Watts-Strogatz small world network', fontsize=TitleFontSize)
ax_BA_cc.set_title('Extended BA scale-free')

ax_BA.set_xlabel('Clustering coefficient', fontsize=LabelFontSize)
ax_WS.set_xlabel('Clustering coefficient', fontsize=LabelFontSize)
ax.set_ylabel('Degree', fontsize=LabelFontSize)
ax_BA.set_ylabel('Degree', fontsize=LabelFontSize)

plt.draw()