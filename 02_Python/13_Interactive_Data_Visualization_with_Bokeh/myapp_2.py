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
from numpy.random import random, normal, lognormal
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
from bokeh.layouts import widgetbox, column
from bokeh.models import Slider, ColumnDataSource, Select







# How to combine Bokeh models into layouts
N = 300
source = ColumnDataSource(data={'x': random(N), 'y': random(N)})

# Create ColumnDataSource: source
# Create plots and widgets
plot = figure()
plot.circle(x='x', y='y', source=source)
menu = Select(options=['uniform', 'normal', 'lognormal'], value='uniform', title='Distribution')
# Add callback to widgets
def callback(attr, old, new):
    if menu.value == 'uniform': f = random
    elif menu.value == 'normal': f = normal
    else: f = lognormal
    source.data={'x': f(size=N), 'y': f(size=N)}
menu.on_change('value', callback)
# Arrange plots and widgets in layouts
layout = column(menu, plot)
curdoc().add_root(layout)


# bokeh serve --show myapp_2.py