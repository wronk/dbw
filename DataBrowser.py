import pdb
"""
Created on Wed Jun 25 10:48:06 2014

@author: rkp

Class for graphically browsing through data point & displaying their 
properties.
"""

import numpy as np
import matplotlib.pyplot as plt

class DataBrowser():
    """Class for DataBrowser."""
    
    def __init__(self,feature0,feature1,data_dict,data_type='edge'):
        """DataBrowser constructor."""
        
        self.f0 = feature0
        self.f1 = feature1
        self.data_dict = data_dict
        self.data_type = data_type
        
    def set_GUI(self,fig,ax,line):
        """Set up the GUI."""
        
        self.lastind = 0
        self.fig = fig
        self.ax = ax
        self.line = line
        
        self.selected, = ax[0].plot([self.f0[0]], [self.f1[0]], 'o', ms=12, 
                                    alpha=0.4, color='yellow', visible=False)
    
    def onclick(self,event):
        """Click event handler."""

        N = len(event.ind)
        if not N: 
            return True
        
        # Get click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        
        distances = np.hypot(x-self.f0[event.ind], y-self.f1[event.ind])
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()
        
    def update(self):
        """Update plot."""
        
        if self.lastind is None: return
        dataind = self.lastind
        
        if self.data_type == 'edge':
            acronym = '%s , %s'%(self.data_dict[dataind]['acronym0'],
                                 self.data_dict[dataind]['acronym1'])
            name = '%s ,\n %s'%(self.data_dict[dataind]['name0'],
                              self.data_dict[dataind]['name1'])
        elif self.data_type == 'node':
            acronym = self.data_dict[dataind]['acronym']
            name = self.data_dict[dataind]['name']
        self.ax[1].cla()
        self.ax[1].text(.1,.9,acronym)
        self.ax[1].text(.1,.6,name)
        
        self.selected.set_visible(True)
        self.selected.set_data(self.f0[dataind], self.f1[dataind])
        
        self.fig.canvas.draw()