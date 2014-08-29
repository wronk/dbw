import pdb
"""
Created on Fri Aug 29 14:02:31 2014

@author: rkp

Functions for plotting area-examples & area statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def scatter_2D(ax,area_dict,feat1,feat2,corr_line=True,**kwargs):
    """Plot two features against each other given a feature dictionary."""
    
    # Get first & second feature
    keys = area_dict.keys()
    feat1_vals = np.array([area_dict[key][feat1] for key in keys])
    feat2_vals = np.array([area_dict[key][feat2] for key in keys])
    
    # Make scatter plot
    ax.scatter(feat1_vals,feat2_vals,**kwargs)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    
    if corr_line:
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(feat1_vals,feat2_vals)
        ax.plot(feat1_vals,slope*feat1_vals+intercept,'b',lw=3)
        print 'R = %.5f'%r_value
        print 'P = %.5f'%p_value