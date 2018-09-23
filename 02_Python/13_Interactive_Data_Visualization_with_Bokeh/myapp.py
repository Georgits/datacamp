# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:34:07 2018

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\13_Interactive_Data_Visualization_with_Bokeh'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\13_Interactive_Data_Visualization_with_Bokeh'
os.chdir(path)

from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show, curdoc
from bokeh.models import CategoricalColorMapper, Slider, ColumnDataSource, HoverTool, Select
from bokeh.layouts import row, column, widgetbox, gridplot
from bokeh.models.widgets import Panel, Tabs
from bokeh.palettes import Spectral6


# Introducing the project dataset
data = pd.read_csv('gapminder_tidy.csv', index_col = 'Year')
data['fertility'] = pd.to_numeric(data['fertility'])
data['life'] = pd.to_numeric(data['life'])
data['population'] = pd.to_numeric(data['population'])
data['child_mortality'] = pd.to_numeric(data['child_mortality'])
data['gdp'] = pd.to_numeric(np.log(data['gdp']))

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()
# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)


# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
    'pop'     : (data.loc[1970].population / 20000000) + 2,
    'region'  : data.loc[1970].region,
    'gdp'     : data.loc[1970].gdp,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)
# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)


# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)
# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)




# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax),
              tools='box_select,lasso_select')
# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')
# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'



# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y

    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
        'gdp'     : data.loc[yr].gdp,
    }
    # Assign new_data to: source.data
    source.data = new_data
    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr
    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])
# Add the HoverTool to the plot
plot.add_tools(hover)

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)
# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)
# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)



# Make a slider object: slider
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)
curdoc().title = 'Gapminder'


