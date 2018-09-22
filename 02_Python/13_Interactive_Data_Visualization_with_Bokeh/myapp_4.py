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


# Synchronize two dropdowns
# Create two dropdown Select widgets: select1, select2
select1 = Select(title='First', options=['A', 'B'], value='A')
select2 = Select(title='Second', options=['1', '2', '3'], value='1')
# Define a callback function: callback
def callback(attr, old, new):
    # If select1 is 'A' 
    if select1.value == 'A':
        # Set select2 options to ['1', '2', '3']
        select2.options = ['1', '2', '3']

        # Set select2 value to '1'
        select2.value = '1'
    else:
        # Set select2 options to ['100', '200', '300']
        select2.options = ['100', '200', '300']

        # Set select2 value to '100'
        select2.value = '100'

# Attach the callback to the 'value' property of select1
select1.on_change('value', callback)
# Create layout and add to current document
layout = widgetbox(select1, select2)
curdoc().add_root(layout)

# bokeh serve --show myapp_4.py