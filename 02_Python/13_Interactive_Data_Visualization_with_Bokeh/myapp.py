# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:34:07 2018

@author: d91067
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:22:49 2017

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\13_Interactive_Data_Visualization_with_Bokeh'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\13_Interactive_Data_Visualization_with_Bokeh'
os.chdir(path)

# Import figure from bokeh.plotting
from bokeh.plotting import figure
# Import output_file and show from bokeh.io
from bokeh.io import output_file, show
# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource
# import the HoverTool
from bokeh.models import HoverTool
#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper
# Import row from bokeh.layouts
from bokeh.layouts import row, column
# # Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot
# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel, Tabs


# Perform necessary imports
from bokeh.io import curdoc
from bokeh.plotting import figure
# Perform the necessary imports
from bokeh.io import curdoc
from bokeh.layouts import widgetbox
from bokeh.models import Slider







# Chapter 3: Building interactive apps with Bokeh
# Using the current document
# Create a new plot: plot
plot = figure()
# Add a line to the plot
plot.line(x = [1,2,3,4,5], y = [2,5,4,6,7])
# Add the plot to the current document
curdoc().add_root(plot)




# Add a single slider
# Create a slider: slider
slider = Slider(title='my slider', start=0, end=10, step=0.1, value=2)
# Create a widgetbox layout: layout
layout = widgetbox(slider)
# Add the layout to the current document
curdoc().add_root(layout)