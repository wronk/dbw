

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import color_scheme
import in_out_plot_config as cf

# Initialize the figure and axes objects
MARKERSIZE = 25.
FONTSIZE = 12.
cf.FONTSIZE = FONTSIZE
ALPHA = 0.5
save = True
save_path = os.getenv('DBW_SAVE_CACHE')

fig = plt.figure(figsize=(7.5, 4), facecolor='w', dpi=300.)
plt.subplots_adjust(bottom=0.15,hspace=0.45,wspace=0.55)

left_main_ax = plt.subplot2grid(cf.subplot_divisions,cf.left_main_location,rowspan=cf.left_main_rowspan,\
                                 colspan=cf.left_main_colspan)

right_main_ax = plt.subplot2grid(cf.subplot_divisions,cf.right_main_location,rowspan=cf.right_main_rowspan,\
                                 colspan=cf.right_main_colspan)

top_margin_ax = plt.subplot2grid(cf.subplot_divisions,cf.top_margin_location,rowspan=cf.top_margin_rowspan,\
                                 colspan=cf.top_margin_colspan,sharex=left_main_ax)

right_margin_ax = plt.subplot2grid(cf.subplot_divisions,cf.right_margin_location,rowspan=cf.right_margin_rowspan,\
                                 colspan=cf.right_margin_colspan,sharey=left_main_ax)


# To get the log axes we need to create another axis on top of our existing ones
top_dummy_ax = top_margin_ax.twinx()
right_dummy_ax = right_margin_ax.twiny()

# create attachment and growht models
G = nx.erdos_renyi_graph(bc.num_brain_nodes, bc.p_brain_edge_directed,
                         directed=True)
# Get in- & out-degree
indeg = np.array([G.in_degree()[node]
                             for node in G])
outdeg = np.array([G.out_degree()[node]
                              for node in G])
deg = indeg + outdeg


# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)

a1 = 1.0
color_scheme.RANDOM = [0.9,0.1,0.0]

# Left main plot (in vs out degree)
left_main_ax.scatter(indeg,outdeg,c=color_scheme.RANDOM,\
                     s=cf.MARKERSIZE,lw=0,alpha=ALPHA)

left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')

left_main_ax.set_xlim([0, 100])
left_main_ax.set_ylim([0, 100])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 101, 25))
left_main_ax.set_yticks(np.arange(0, 101, 25))
left_main_ax.legend(loc='best')
left_main_ax.text(120,120,'a',fontsize=cf.FONTSIZE+2,fontweight='bold')


# Top marginal (in-degree)
top_margin_ax.hist(indeg,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.RANDOM,normed=True,stacked=True)
#top_margin_ax.plot(indeg_x,indeg_y,linestyle='-',lw=4,color='k')

# This is for the log-axis
indeg_hist = np.histogram(indeg,bins=cf.OUTDEGREE_BINS)
indeg_x = indeg_hist[1][0:len(indeg_hist[0])]
indeg_y = indeg_hist[0]
indeg_y = indeg_y/float(indeg_y.sum())

top_dummy_ax.plot(indeg_x,indeg_y,linestyle='-',lw=2,color='b')
top_dummy_ax.yaxis.tick_right()
top_dummy_ax.yaxis.set_label_position('right')
top_dummy_ax.set_yscale('log')

top_margin_ax.set_yticks([0,0,5,1.0])
top_margin_ax.set_ylim([0,1.0])


# Right marginal (out-degree)
right_margin_ax.hist(outdeg,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.RANDOM,orientation='horizontal',normed=True,stacked=True)

# This is for the log-axis
outdeg_hist = np.histogram(outdeg,bins=cf.OUTDEGREE_BINS)
outdeg_x = outdeg_hist[1][0:len(outdeg_hist[0])]
outdeg_y = outdeg_hist[0]
outdeg_y = outdeg_y/float(outdeg_y.sum())

right_dummy_ax.plot(outdeg_y,outdeg_x,linestyle='-',lw=2,color='b')
right_dummy_ax.xaxis.tick_top()
right_dummy_ax.xaxis.set_label_position('top')
right_dummy_ax.set_xscale('log')

top_margin_ax.set_yticks([0,0.05,0.1])
top_margin_ax.set_ylim([0,0.1])


plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),visible=False)

right_margin_ax.set_xticks([0,0.05,0.1])
right_margin_ax.set_xlim([0,0.1])


# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg, percent_indeg, s=cf.MARKERSIZE, lw=0,
            c=color_scheme.RANDOM,alpha=ALPHA)

right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_ylim([0., 1.05])
right_main_ax.set_xticks(np.arange(0, 151, 50))
right_main_ax.text(1., 1.2, 'b', fontsize=cf.FONTSIZE+2, fontweight='bold',
                   transform=right_main_ax.transAxes, ha='right')


top_margin_ax.set_ylabel('$P(K_\mathrm{in}=k)$',va='baseline')
right_margin_ax.set_xlabel('$P(K_\mathrm{out}=k)$',va='top')


for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax,top_dummy_ax,right_dummy_ax]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)



top_lin_ticks = top_margin_ax.get_yticklabels()
right_lin_ticks = right_margin_ax.get_xticklabels()

top_log_ticks = top_dummy_ax.get_yticklabels()
right_log_ticks = right_dummy_ax.get_xticklabels()

for tick in top_lin_ticks+right_lin_ticks:
    tick.set_color('k')
    
for tick in top_log_ticks+right_log_ticks:
    tick.set_color('blue')
    tick.set_fontsize(7.5)

for tick in top_log_ticks:
    pos = tick.get_position()
    tick.set_position((1.0,pos[1]))

for tick in right_log_ticks:
    pos = tick.get_position()
    tick.set_position((pos[0],1.0))

for tick in right_log_ticks+right_lin_ticks:
    tick.set_rotation(270)

fig.subplots_adjust(left=0.125, top=0.925, right=0.95, bottom=0.225)

if save:
    fig.savefig(os.path.join(save_path, 'figure_S1.png'), dpi=300)
    fig.savefig(os.path.join(save_path, 'figure_S1.pdf'), dpi=300)
plt.show(block=False)
