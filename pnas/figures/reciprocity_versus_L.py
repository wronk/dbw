import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize
from extract.brain_graph import binary_directed as brain_graph

from brain_constants import *
from network_compute import reciprocity

import graph_tools.auxiliary as aux_tools

run = False

Ls = np.linspace(0.1,1.7,11)
N = 20
if run:

    modelReciprocity = np.zeros([len(Ls),N])
    
    for j,L in enumerate(Ls):
        for n in range(N):
            G, A, labels = biophysical_model(N=num_brain_nodes,
                                         N_edges=num_brain_edges_directed,
                                         L=L,
                                         gamma=1.)
            recip = reciprocity(A)
            modelReciprocity[j,n] = recip
            print "Reciprocity: " + str(np.mean(modelReciprocity[:,n])) + "(" + str(np.std(modelReciprocity[:,n])) + ")"




meanReciprocity = np.mean(modelReciprocity,axis=1)

G,A,labels = brain_graph()
brainReciprocity = reciprocity(A)

configReciprocity = np.zeros([20,1])
nodes = G.nodes()
inDeg = G.in_degree(); inDeg = [inDeg[node] for node in nodes]
outDeg = G.out_degree(); outDeg = [outDeg[node] for node in nodes]
for j in range(20):
    G_config = nx.directed_configuration_model(inDeg,outDeg)
    A_config = nx.adjacency_matrix(G_config)
    A_configInt = np.zeros(A_config.shape,dtype=int)
    
    A_configInt[A_config < 0.5] = 0
    A_configInt[A_config > 0.5] = 1
    
    configReciprocity[j] = reciprocity(A_configInt)



from extract.brain_graph import binary_directed as brain_graph
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize
import matplotlib

#matplotlib.rc('axes',edgecolor='white')
fig,axs = plt.subplots(1,facecolor='white',edgecolor='white')
meanReciprocity = np.mean(modelReciprocity,axis=1)[0:len(Ls)-1]
stdReciprocity = np.std(modelReciprocity,axis=1)[0:len(Ls)-1]
Ls = Ls[0:len(Ls)-1]
axs.plot(Ls,meanReciprocity,linewidth=2,color='k')
axs.fill_between(Ls,meanReciprocity-stdReciprocity,meanReciprocity+stdReciprocity,\
                 facecolor='k',alpha=0.2,antialiased=True,linewidth=3,linestyle='-',
                 edgecolor='k')


G,A,labels = brain_graph()
brainReciprocity = reciprocity(A)

axs.plot([0,2],[brainReciprocity,brainReciprocity],linewidth=4,color='red',linestyle='--')

configX = [0,2]; configY = [np.mean(configReciprocity),np.mean(configReciprocity)]
axs.plot(configX,configY,linewidth=2,color='b')
axs.fill_between(configX,configY-np.std(configReciprocity),configY+np.std(configReciprocity),\
                 facecolor='b',alpha=0.2,antialiased=True,linewidth=3,linestyle=':',\
                 edgecolor='b')


xticks = np.arange(0,2,0.25)
yticks = [0,0.15,0.3,0.45,0.6]

axs.set_xlabel('L (mm)',fontsize=26,color='k')
axs.set_ylabel('Reciprocity',fontsize=26,color='k')
axs.set_xticks(xticks)
axs.set_yticks(yticks)
axs.set_xticklabels(xticks,fontsize=22,color='k')
axs.set_yticklabels(yticks,fontsize=22,color='k')

axs.set_xlim([0,Ls[len(Ls)-2]+Ls[0]])
axs.set_ylim([0,max(meanReciprocity)+min(meanReciprocity)])

leg = axs.legend(['PGPA model', 'Allen Atlas', 'Shuffled Atlas'],prop={'size':22})

plt.show(block=False)
