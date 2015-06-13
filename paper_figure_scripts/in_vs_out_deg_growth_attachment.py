
import numpy as np
import matplotlib.pyplot as plt

from random_graph.binary_directed import biophysical_indegree, biophysical_reverse_outdegree

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import color_scheme
import in_out_plot_config as cf


### Set up parameters for the subplots
subplot_divisions = (4,8)

top_margin_colspan = 3
top_margin_rowspan = 1
top_margin_location = (0,0)

right_margin_colspan = 1
right_margin_rowspan = 3
right_margin_location = (1,3)

left_main_colspan = 3
left_main_rowspan = 3
left_main_location = (1,0)

right_main_colspan = 3
right_main_rowspan = 3
right_main_location = (1,5)


# Initialize the figure and axes objects
k = 1.25
fig = plt.figure(figsize=(10*k,4*k),facecolor='w')
plt.subplots_adjust(bottom=0.15,hspace=0.45,wspace=0.45)

left_main_ax = plt.subplot2grid(subplot_divisions,left_main_location,rowspan=left_main_rowspan,\
                                 colspan=left_main_colspan)

right_main_ax = plt.subplot2grid(subplot_divisions,right_main_location,rowspan=right_main_rowspan,\
                                 colspan=right_main_colspan)

top_margin_ax = plt.subplot2grid(subplot_divisions,top_margin_location,rowspan=top_margin_rowspan,\
                                 colspan=top_margin_colspan,sharex=left_main_ax)

right_margin_ax = plt.subplot2grid(subplot_divisions,right_margin_location,rowspan=right_margin_rowspan,\
                                 colspan=right_margin_colspan,sharey=left_main_ax)



# create attachment and growht models
Gattachment, _, _ = biophysical_indegree(N=bc.num_brain_nodes,
                                         N_edges=bc.num_brain_edges_directed,
                                         L=np.inf, gamma=1.)

Ggrowth, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=np.inf, gamma=1.)

# Get in- & out-degree
indeg_attachment = np.array([Gattachment.in_degree()[node]
                             for node in Gattachment])
outdeg_attachment = np.array([Gattachment.out_degree()[node]
                              for node in Gattachment])
deg_attachment = indeg_attachment + outdeg_attachment

indeg_growth = np.array([Ggrowth.in_degree()[node] for node in Ggrowth])
outdeg_growth = np.array([Ggrowth.out_degree()[node] for node in Ggrowth])
deg_growth = indeg_growth + outdeg_growth

# Calculate proportion in degree
percent_indeg_attachment = indeg_attachment / deg_attachment.astype(float)
percent_indeg_growth = indeg_growth / deg_growth.astype(float)

legend = ['Preferential\nattachment', 'Preferential\ngrowth']
a1 = 0.6


# Left main plot (in vs out degree)
left_main_ax.scatter(indeg_attachment,outdeg_attachment,c=color_scheme.PREFATTACHMENT,\
                     s=cf.MARKERSIZE,lw=0,alpha=a1,label=legend[0])
left_main_ax.scatter(indeg_growth,outdeg_growth,c=color_scheme.PREFGROWTH,\
                     s=cf.MARKERSIZE,lw=0,alpha=a1,label=legend[1])
left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')

left_main_ax.set_xlim([0, 100])
left_main_ax.set_ylim([0, 100])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 101, 25))
left_main_ax.set_yticks(np.arange(0, 101, 25))
#left_main_ax.legend(loc='best')

# Top marginal (in-degree)
top_margin_ax.hist(indeg_attachment,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.PREFATTACHMENT,alpha=a1,label='Preferential attachment')
top_margin_ax.hist(indeg_growth,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.PREFGROWTH,alpha=a1,label='Preferential growth')
leg=top_margin_ax.legend(loc='upper right')

# Right marginal (out-degree)
right_margin_ax.hist(outdeg_attachment,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.PREFATTACHMENT,alpha=a1,orientation='horizontal')
right_margin_ax.hist(outdeg_growth,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.PREFGROWTH,alpha=a1,orientation='horizontal')

plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),visible=False)

top_margin_ax.set_yticks([0,60,120])
top_margin_ax.set_ylim([0,120])
right_margin_ax.set_xticks([0,60,120])
right_margin_ax.set_xlim([0,120])


# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg_attachment, percent_indeg_attachment, s=cf.MARKERSIZE, lw=0,
            c=color_scheme.PREFATTACHMENT, alpha=al)
right_main_ax.scatter(deg_growth, percent_indeg_growth, s=cf.MARKERSIZE, lw=0,
            c=color_scheme.PREFGROWTH, alpha=al)
right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_ylim([0., 1.05])
right_main_ax.set_xticks(np.arange(0, 151, 50))


for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)


plt.show(block=False)
