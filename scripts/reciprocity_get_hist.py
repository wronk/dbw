# Author: sid
# Script to generate N graphs with length parameter L
# Plot a resulting histogram of these distributions
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize
from extract.brain_graph import binary_directed as brain_graph

from brain_constants import *
from network_compute import reciprocity
import graph_tools.auxiliary as aux_tools

# Number of iterations per L
N = 250

# Specify length coefficients
Ls = [np.inf,1,0.75]

modelReciprocity = np.zeros([len(Ls),N])


for n in range(N):
    for j,L in enumerate(Ls):
        G, A, labels = biophysical_model(N=num_brain_nodes,
                                     N_edges=num_brain_edges_directed,
                                     L=L,
                                     gamma=1.)
        # Compute reciprocity
        modelReciprocity[j,n] = reciprocity(A)
    

# Plot histograms
matplotlib.rc('axes',edgecolor='white')
fig,axs = plt.subplots(1,facecolor='white',edgecolor='white')

bins = np.linspace(0,0.2,41)
model0 = np.histogram(modelReciprocity[0,:],bins)
model1 = np.histogram(modelReciprocity[1,:],bins)

axs.hist(modelReciprocity[0,:],bins)
axs.hist(modelReciprocity[1,:],bins)
axs.hist(modelReciprocity[2,:],bins)
xticks = [0,0.05,0.1,0.15,0.2]
yticks = [0,30,60]
axs.set_xlabel('Reciprocity',fontsize=48,color='white')
axs.set_ylabel('Count',fontsize=48,color='white')

axs.set_xticks(xticks)
axs.set_xticklabels(xticks,fontsize=36,color='white')

axs.set_yticks(yticks)
axs.set_yticklabels(yticks,fontsize=36,color='white')


G,A,labels = brain_graph()
brainReciprocity = reciprocity(A)

axs.plot([brainReciprocity,brainReciprocity],[0,70], color='red',linewidth=4)


fig.patch.set(alpha=0)
axs.patch.set(alpha=0)

myleg = axs.legend(['Reciprocity in Allen Atlas','L=0.8','L=0.4', 'L=0'],\
           prop={'size':32},loc='upper left',fancybox=True)
text1,text2,text3,text4 = myleg.get_texts()
text1.set_color('white'); text2.set_color('white'); text3.set_color('white')
text4.set_color('white')
myleg.get_frame().set(alpha=0)

plt.show(block=False)
