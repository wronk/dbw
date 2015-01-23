"""
Created on Fri Jan 23 13:39:04 2015

@author: rkp

Code for quickly changing graphical settings.
"""

import numpy as np
import matplotlib.pyplot as plt

def set_all_text_fontsizes(ax, fontsize=16):
    """Set fontsize of all text elements in an axis."""
    
    text_items = []
    text_items += [ax.title, ax.xaxis.label, ax.yaxis.label]
    text_items += ax.get_xticklabels()
    text_items += ax.get_yticklabels()
    
    for text in text_items:
        text.set_fontsize(fontsize)