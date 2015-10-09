"""
Created on Sun May 17, 2015

@author: wronk

In and outdegree plotting configuration parameters.
"""

import numpy as np

FIGSIZE = (20, 9)
FONTSIZE = 27
MARKERSIZE = 50
BINWIDTH = 4
FACECOLOR = 'w'
LABELCOLOR = 'k'
TICKSIZE = 4.

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

INDEGREE_BINS = np.linspace(0, 150, 41)
OUTDEGREE_BINS = np.linspace(0, 150, 41)
