import os
import numpy as np
import matplotlib.pyplot as plt

from random_graph.binary_directed import (biophysical_indegree,
                                          biophysical_reverse_outdegree)

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc

import color_scheme
import in_out_plot_config as cf

save = True
if save:
    save_path = os.environ['DBW_SAVE_CACHE']

MARKERSIZE = 25.
FONTSIZE = 12.
ALPHA = 0.5

fig = plt.figure(figsize=(7.5, 4), facecolor='w', dpi=300.)
plt.subplots_adjust(hspace=0.45, wspace=0.45)

left_main_ax = plt.subplot2grid(cf.subplot_divisions, cf.left_main_location,
                                rowspan=cf.left_main_rowspan,
                                colspan=cf.left_main_colspan)

right_main_ax = plt.subplot2grid(cf.subplot_divisions, cf.right_main_location,
                                 rowspan=cf.right_main_rowspan,
                                 colspan=cf.right_main_colspan)

top_margin_ax = plt.subplot2grid(cf.subplot_divisions, cf.top_margin_location,
                                 rowspan=cf.top_margin_rowspan,
                                 colspan=cf.top_margin_colspan,
                                 sharex=left_main_ax)

right_margin_ax = plt.subplot2grid(cf.subplot_divisions,
                                   cf.right_margin_location,
                                   rowspan=cf.right_margin_rowspan,
                                   colspan=cf.right_margin_colspan,
                                   sharey=left_main_ax)

# create attachment and growth models
Ggrowth, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=np.inf, gamma=1.)
Gattachment, _, _ = biophysical_indegree(N=bc.num_brain_nodes,
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

# Left main plot (in vs out degree)
left_main_ax.scatter(indeg_growth, outdeg_growth, c=color_scheme.SRCGROWTH,
                     s=MARKERSIZE, lw=0, alpha=ALPHA, zorder=3)
left_main_ax.scatter(indeg_attachment, outdeg_attachment,
                     c=color_scheme.TARGETATTRACTION,
                     s=MARKERSIZE, lw=0, alpha=ALPHA)
left_main_ax.set_xlabel('In-degree')
left_main_ax.set_ylabel('Out-degree')

left_main_ax.set_xlim([0, 125])
left_main_ax.set_ylim([0, 125])
left_main_ax.set_aspect('auto')
left_main_ax.set_xticks(np.arange(0, 121, 40))
left_main_ax.set_yticks(np.arange(0, 121, 40))
left_main_ax.text(150, 150, 'a', fontsize=FONTSIZE + 2, fontweight='bold')

# Top marginal (in-degree)
top_margin_ax.hist(indeg_attachment, bins=cf.OUTDEGREE_BINS,
                   histtype='stepfilled', color=color_scheme.TARGETATTRACTION,
                   alpha=ALPHA, label='Target attraction', normed=True,
                   stacked=True)
top_margin_ax.hist(indeg_growth, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                   color=color_scheme.SRCGROWTH, alpha=ALPHA,
                   label='Source growth', normed=True, stacked=True)

# Right marginal (out-degree)
right_margin_ax.hist(outdeg_attachment, bins=cf.OUTDEGREE_BINS,
                     histtype='stepfilled',
                     color=color_scheme.TARGETATTRACTION, alpha=ALPHA,
                     orientation='horizontal', normed=True, stacked=True)
right_margin_ax.hist(outdeg_growth, bins=cf.OUTDEGREE_BINS,
                     histtype='stepfilled', color=color_scheme.SRCGROWTH,
                     alpha=ALPHA, orientation='horizontal', normed=True,
                     stacked=True)
for tick in right_margin_ax.get_xticklabels():
    tick.set_rotation(270)

plt.setp(right_margin_ax.get_yticklabels() + top_margin_ax.get_xticklabels(),
         visible=False)

top_margin_ax.set_yticks([0, 0.05, 0.1])
top_margin_ax.set_ylim([0, 0.1])
right_margin_ax.set_xticks([0, 0.05, 0.1])
right_margin_ax.set_xlim([0, 0.1025])

top_margin_ax.set_ylabel('$P(K_\mathrm{in} = k)$', va='baseline')
right_margin_ax.set_xlabel('$P(K_\mathrm{out} = k)$', va='top')

# Right main plot (proportion in vs total degree)
right_main_ax.scatter(deg_growth, percent_indeg_growth, s=MARKERSIZE, lw=0,
                      c=color_scheme.SRCGROWTH, alpha=ALPHA,
                      label='Source growth', zorder=3)
right_main_ax.scatter(deg_attachment, percent_indeg_attachment,
                      s=MARKERSIZE, lw=0, c=color_scheme.TARGETATTRACTION,
                      alpha=ALPHA, label='Target attraction')
right_main_ax.xaxis.set_major_locator(plt.MaxNLocator(4))
right_main_ax.set_yticks(np.arange(0, 1.1, .25))
right_main_ax.set_xticks(np.arange(0, 151, 50))
right_main_ax.set_xlabel('Total degree (in + out)')
right_main_ax.set_ylabel('Proportion in-degree')
right_main_ax.text(1., 1.2, 'b', fontsize=FONTSIZE + 2, fontweight='bold',
                   transform=right_main_ax.transAxes, ha='right')
right_main_ax.set_xlim([0., 150.])
right_main_ax.set_ylim([-0.025, 1.025])
right_main_ax.legend(loc=(-0.35, 1.12), prop={'size': 12})

for temp_ax in [left_main_ax, right_main_ax, top_margin_ax, right_margin_ax]:
    set_all_text_fontsizes(temp_ax, FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    temp_ax.tick_params(width=1.)

fig.subplots_adjust(left=0.125, top=0.925, right=0.95, bottom=0.225)
if save:
    fig.savefig(os.path.join(save_path, 'figure_4.png'), dpi=150)
    fig.savefig(os.path.join(save_path, 'figure_4.pdf'), dpi=300)
plt.show(block=False)
