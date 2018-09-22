# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:35:30 2018

@author: d91067
"""

from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.plotting import figure
from numpy.random import random
# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle

import pandas as pd
import numpy as np


N = 300
x = random(N) *20
y = random(N)

source = ColumnDataSource(data={'x': x, 'y': np.sin(x) + np.random.random(N)})
plot = figure()
plot.circle(x= 'x', y='y', source=source)

# Create a Button with label 'Update Data'
button = Button(label = 'Update Data')
# Define an update callback with no arguments: update
def update():
    # Compute new y values: y
    y = np.sin(x) + np.random.random(N)
    # Update the ColumnDataSource data dictionary
    source.data = {'x': x, 'y': y}
# Add the update callback to the button
button.on_click(update)


# Arrange plots and widgets in layouts
layout = column(button, plot)



# Add a Toggle: toggle
toggle = Toggle(button_type='success', label = 'Toggle button')
# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])
# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])


# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(column(layout, widgetbox(toggle, checkbox, radio)))   

#curdoc().add_root(layout)

# bokeh serve --show myapp_5.py



