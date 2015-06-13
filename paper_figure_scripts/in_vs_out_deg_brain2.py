
import numpy as np
import matplotlib.pyplot as plt


from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

from extract.brain_graph import binary_directed

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
k = 2.0

cf.MARKERSIZE *= k
cf.TICKSIZE *= k
cf.FONTSIZE *= 1.5

fig = plt.figure(figsize=(10*k,4*k),facecolor='w')
plt.subplots_adjust(bottom=0.15,hspace=0.45,wspace=0.55)

left_main_ax = plt.subplot2grid(subplot_divisions,left_main_location,rowspan=left_main_rowspan,\
                                 colspan=left_main_colspan)

right_main_ax = plt.subplot2grid(subplot_divisions,right_main_location,rowspan=right_main_rowspan,\
                                 colspan=right_main_colspan)

top_margin_ax = plt.subplot2grid(subplot_divisions,top_margin_location,rowspan=top_margin_rowspan,\
                                 colspan=top_margin_colspan,sharex=left_main_ax)

right_margin_ax = plt.subplot2grid(subplot_divisions,right_margin_location,rowspan=right_margin_rowspan,\
                                 colspan=right_margin_colspan,sharey=left_main_ax)


# To get the log axes we need to create another axis on top of our existing ones
top_dummy_ax = top_margin_ax.twinx()
right_dummy_ax = right_margin_ax.twiny()
               

#right_dummy_ax = plt.subplot2grid(subplot_divisions,right_margin_location,rowspan=right_margin_rowspan,\
#                                 colspan=right_margin_colspan,sharey=left_main_ax,frameon=False)

# create attachment and growht models
G, _, _ = binary_directed()          

# Get in- & out-degree
indeg = np.array([G.in_degree()[node]
                             for node in G])
outdeg = np.array([G.out_degree()[node]
                              for node in G])
deg = indeg + outdeg_attachment


# Calculate proportion in degree
percent_indeg = indeg / deg_attachment.astype(float)

a1 = 1.0


# Left main plot (in vs out degree)
left_main_ax.scatter(indeg,outdeg,c=color_scheme.ATLAS,\
                     s=cf.MARKERSIZE,lw=0)

left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')

left_main_ax.set_xlim([0, 100])
left_main_ax.set_ylim([0, 100])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 101, 25))
left_main_ax.set_yticks(np.arange(0, 101, 25))
left_main_ax.legend(loc='best')


# Top marginal (in-degree)
top_margin_ax.hist(indeg,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.ATLAS)
top_margin_ax.plot(indeg_x,indeg_y,linestyle='-',lw=4,color='k')

# This is for the log-axis
indeg_hist = np.histogram(indeg,bins=cf.OUTDEGREE_BINS)
indeg_x = indeg_hist[1][0:len(indeg_hist[0])]
indeg_y = indeg_hist[0]
indeg_y = indeg_y/float(indeg_y.sum())

top_dummy_ax.plot(indeg_x,indeg_y,linestyle='-',lw=3,color='b')
top_dummy_ax.yaxis.tick_right()
top_dummy_ax.yaxis.set_label_position('right')
top_dummy_ax.set_yscale('log')

top_margin_ax.set_yticks([0,60,120])
top_margin_ax.set_ylim([0,120])


# Right marginal (out-degree)
right_margin_ax.hist(outdeg,bins=cf.OUTDEGREE_BINS,histtype='stepfilled',\
                     color=color_scheme.ATLAS,orientation='horizontal')

# This is for the log-axis
outdeg_hist = np.histogram(outdeg,bins=cf.OUTDEGREE_BINS)
outdeg_x = outdeg_hist[1][0:len(outdeg_hist[0])]
outdeg_y = outdeg_hist[0]
outdeg_y = outdeg_y/float(outdeg_y.sum())

right_dummy_ax.plot(outdeg_y,outdeg_x,linestyle='-',lw=3,color='b')
right_dummy_ax.xaxis.tick_top()
right_dummy_ax.xaxis.set_label_position('top')
right_dummy_ax.set_xscale('log')

top_margin_ax.set_yticks([0,60,120])
top_margin_ax.set_ylim([0,120])


plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),visible=False)

right_margin_ax.set_xticks([0,60,120])
right_margin_ax.set_xlim([0,120])


# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg_attachment, percent_indeg, s=cf.MARKERSIZE, lw=0,
            c=color_scheme.ATLAS)

right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_ylim([0., 1.05])
right_main_ax.set_xticks(np.arange(0, 151, 50))


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
    tick.set_fontsize(16)

for tick in right_log_ticks+right_lin_ticks:
    tick.set_rotation(270)

plt.show(block=False)
