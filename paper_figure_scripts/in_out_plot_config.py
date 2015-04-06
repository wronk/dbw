"""
Created on Mon Feb 23 10:16:12 2015

@author: rkp, wronk

In and outdegree plotting configuration parameters.
"""

import numpy as np

FIGSIZE = (16, 7)
FONTSIZE = 20
MARKERSIZE = 25
BINWIDTH = 4
FACECOLOR = 'white'
LABELCOLOR = 'black'
TICKSIZE = 2.

SUBPLOT_DIVISIONS = (1, 3)
AX0_LOCATION = (0, 0)
AX0_COLSPAN = 2
AX1_LOCATION = (0, 2)
AX1_COLSPAN = 1

IN_OUT_SCATTER_XLIM = (0, 55)
IN_OUT_SCATTER_YLIM = (0, 150)

INDEGREE_BINS = np.linspace(0, 55, 20)
OUTDEGREE_BINS = np.linspace(0, 150, 20)
