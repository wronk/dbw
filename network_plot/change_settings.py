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

def set_all_colors(ax, color):
    """Set colors on all parts of axis."""

    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.spines['right'].set_color(color)

    ax.tick_params(axis='x', color=color)
    ax.tick_params(axis='y', color=color)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color(color)

    ax.title.set_color(color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)

