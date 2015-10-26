from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools
import networkx as nx
import os
import pandas as pd
import pickle

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs

from brain_constants import *
from network_compute import reciprocity
import color_scheme

data_dir=os.getenv('DBW_DATA_DIRECTORY')
pickle_file = data_dir + '/growth_degs.pickle'
n_runs = 1

# Set plot params
labelsize = 10
ticksize = 9
legendsize = 8

fig,axs = plt.subplots(1,3,facecolor='white',edgecolor='white',figsize=(7.5,2.3),dpi=300.)
fig.subplots_adjust(bottom=0.15,wspace=0.42,hspace=0.35)


graphs = ['sgpa','sg','brain']
### Compute/load distances ###
if not os.path.isfile(pickle_file):    
    # init to empty lists
    recip_distances = {};
    nonrecip_distances = {}
    for graph in graphs:
        recip_distances[graph] = []
        nonrecip_distances[graph] = []



    for i in range(n_runs):
        print "iteration %i" % (i+1)
        G_sgpa, A_sgpa, _ = biophysical_model(N=num_brain_nodes,
                                            N_edges=num_brain_edges_directed,
                                            L=.75, gamma=1., brain_size=[7.,7.,7.])

        G_sg, A_sg, _ = biophysical_model(N=num_brain_nodes,
                                            N_edges=num_brain_edges_directed,
                                            L=np.inf, gamma=1., brain_size=[7.,7.,7.])

        # Only run brain the first time
        if i == 0:
            G_brain, A_brain, labels = brain_graph()
            centroids = G_brain.centroids
            label_mapping = {k:labels[k] for k in range(len(labels))}
            G_brain = nx.relabel_nodes(G_brain,label_mapping)
            G_brain.centroids = centroids
            Gs = [G_sgpa,G_sg,G_brain]
            As = [A_sgpa,A_sg,A_brain]
        else:
            Gs = [G_sgpa,G_sg]
            As = [A_sgpa,A_sg]
                
        for i_G,G in enumerate(Gs):
            centroids = G.centroids
            A = As[i_G]

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
                    recip_names.append(edge)
                else:
                    nonrecip_names.append(edge)

            # append the appropriate edges to the correct list
            nonrecip_distances[graphs[i_G]].extend([edges[k] for k in nonrecip_names])
            recip_distances[graphs[i_G]].extend([edges[k] for k in recip_names])


    pickle.dump([nonrecip_distances,recip_distances],open(pickle_file,'wb'))

        

else:    
    with open(pickle_file,'r') as f:
        distances = pickle.load(f)
        nonrecip_distances = distances[0]
        recip_distances = distances[1]



### Compute reciprocity ###
########################
### MOUSE CONNECTOME ###
########################
# load brain graph, adjacency matrix, and labels
G_brain, A_brain, labels_brain = brain_graph()
brain_reciprocity = reciprocity(A_brain)


##################
### SGPA MODEL ###
##################
Ls = np.linspace(0,2,21)
# Load the saved reciprocity values
model_reciprocity = pd.read_csv(data_dir+'/reciprocity2.csv',index_col=0)
mean_reciprocity = np.mean(model_reciprocity,axis=1)
std_reciprocity = np.std(model_reciprocity,axis=1)


####################
### CONFIG MODEL ###
####################
label_mapping = {k:labels_brain[k] for k in range(len(labels_brain))}
G_remap= nx.relabel_nodes(G_brain,label_mapping)
n_repeats = 20
config_reciprocity = np.zeros([n_repeats,1])

# First get lists of in and out deg to feed to config function
nodes = G_remap.nodes()
indeg = G_remap.in_degree(); indeg = [indeg[node] for node in nodes]
outdeg = G_remap.out_degree(); outdeg = [outdeg[node] for node in nodes]

# Loop over 20 (to get a good mean value and SD estimate)
for j in range(n_repeats):
    G_config = nx.directed_configuration_model(indeg,outdeg)
    A_config = nx.adjacency_matrix(G_config).toarray()
    config_reciprocity[j] = reciprocity(A_config)


#####################
### GENERATE PLOT ###
#####################
bins = np.linspace(0,10,31)
dx = bins[1]-bins[0]

### Figure 5a ###
dist_yticks = [0,0.15,0.3,0.45,0.6,0.75]
dist_xticks = [0,2.5,5.0,7.5,10.0]
axs[0].hist(nonrecip_distances['brain'],bins,normed=True,facecolor='b',alpha=0.5)
axs[0].hist(recip_distances['brain'],bins,normed=True,facecolor='g',alpha=0.5)

axs[0].set_xticks(dist_xticks)
axs[0].set_xticklabels(dist_xticks,fontsize=labelsize)
axs[0].set_yticks(dist_yticks)
axs[0].set_yticklabels(dist_yticks,fontsize=labelsize)
axs[0].set_xlabel('Distance (mm)', fontsize=labelsize)
axs[0].set_ylabel('Probability density', fontsize=labelsize)
axs[0].set_xlim([0,max(bins)])
axs[0].set_title('Connectome',fontsize=labelsize+2)
axs[0].legend(['Nonreciprocal','Reciprocal'],prop={'size':legendsize})


### Figure 5b ###
# First plot mean reciprocity and fill in +/- 1 SD
axs[1].plot(Ls[1:],mean_reciprocity[1:],lw=2,color=color_scheme.PGPA)
axs[1].fill_between(Ls[1:],mean_reciprocity[1:]-std_reciprocity[1:],
                    mean_reciprocity[1:]+std_reciprocity[1:],
                    facecolor=color_scheme.PGPA,
                    alpha=0.2,antialiased=True,lw=2,linestyle='-',
                    edgecolor=color_scheme.PGPA)

# Plot brain reciprocity
axs[1].plot([0,2],[brain_reciprocity,brain_reciprocity],lw=2,
            color=color_scheme.ATLAS,linestyle='--',dashes=(4,4))

# Plot configuration reciprocity
configX = [0,2]; configY = [np.mean(config_reciprocity),
                            np.mean(config_reciprocity)]
axs[1].plot(configX,configY,lw=2,color=color_scheme.CONFIG)
axs[1].fill_between(configX,configY-np.std(config_reciprocity),
                    configY+np.std(config_reciprocity),
                    facecolor=color_scheme.CONFIG,alpha=0.2,antialiased=True,
                    lw=2,linestyle=':', edgecolor=color_scheme.CONFIG)


# Set ticks, labels, and font sizes
xticks = np.arange(0,2.5,0.5)
yticks = [0,0.1,0.2,0.3]
axs[1].set_xlabel('L (mm)',fontsize=labelsize,color='k')
axs[1].set_ylabel('Reciprocity coefficient',fontsize=labelsize,color='k')
axs[1].set_xticks(xticks)
axs[1].set_yticks(yticks)
axs[1].set_xticklabels(xticks,fontsize=labelsize,color='k')
axs[1].set_yticklabels(yticks,fontsize=labelsize,color='k')
axs[1].set_xlim([Ls[1],2])
axs[1].set_ylim([0,0.3])
leg = axs[1].legend(['SGPA model', 'Connectome', 'SG/Random'],
                    prop={'size':legendsize})
axs[1].set_title('Length constant fit',fontsize=labelsize+2)




# binned reciprocal distances for SGPA model
binned_rd_sgpa = np.histogram(recip_distances['sgpa'],bins,normed=True)

binned_recip_distances = np.histogram(recip_distances,bins,normed=True)
dx = bins[1]-bins[0]
proportionRecip = binned_recip_distances[0]

axs[2].hist(nonrecip_distances['sgpa'],bins,normed=True,facecolor='b',alpha=0.5)
axs[2].hist(recip_distances['sgpa'],bins,normed=True,facecolor='g',alpha=0.5)

binned_rd_sg,_ = np.histogram(recip_distances['sg'],bins,normed=True)
binned_nrd_sg,_ = np.histogram(nonrecip_distances['sg'],bins,normed=True)

axs[2].plot(bins[0:len(bins)-1]+dx/2.,binned_nrd_sg,'-',lw=2,c='r')
axs[2].plot(bins[0:len(bins)-1]+dx/2.,binned_rd_sg,'-',lw=1,c='k')

axs[2].set_xticks(dist_xticks)
axs[2].set_xticklabels(dist_xticks,fontsize=labelsize)
axs[2].set_yticks(dist_yticks)
axs[2].set_yticklabels([])
axs[2].set_xlabel('Distance (mm)', fontsize=labelsize)
axs[2].set_ylabel('Probability density', fontsize=labelsize)
axs[2].set_xlim([0,10.0])
axs[2].set_title('Growth models',fontsize=labelsize+2)
leg=axs[2].legend(['SG: Nonreciprocal', 'SG: Reciprocal', 'SGPA: Nonreciprocal','SGPA: Reciprocal']\
               ,prop={'size':legendsize},bbox_to_anchor=(1.2,1.0))

leg.draggable()
# Finally add labels
axs[0].text(10*0.05,0.75*0.925,'a',fontsize=labelsize,fontweight='bold')
axs[1].text(0.1+2*0.05,0.3*0.925,'b',fontsize=labelsize,fontweight='bold')
axs[2].text(10*0.05,0.75*0.925,'c',fontsize=labelsize,fontweight='bold')

fig.subplots_adjust(left=0.125, top=0.9, right=0.95, bottom=0.225)

fig_dir = os.getenv('DBW_SAVE_CACHE')
fig.savefig(fig_dir+'fig5.pdf')
plt.draw()
