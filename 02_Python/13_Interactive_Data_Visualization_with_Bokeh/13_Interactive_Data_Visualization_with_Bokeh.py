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


# Chapter 1:  Basic plotting with Bokeh
# A simple scatter plot
literacy_birth_rate = pd.read_csv('literacy_birth_rate.csv')
literacy_birth_rate = literacy_birth_rate.iloc[:162,:]
fertility = pd.to_numeric(literacy_birth_rate['fertility'])
female_literacy = pd.to_numeric(literacy_birth_rate['female literacy'])

fertility_latinamerica = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'LAT']['fertility'])
female_literacy_latinamerica = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'LAT']['female literacy'])

fertility_africa = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'AF']['fertility'])
female_literacy_africa = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'AF']['female literacy'])

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)
# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')
# Display the plot
show(p)



# A scatter plot with different shapes
# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')
# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)
# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)
# Specify the name of the file
output_file('fert_lit_separate.html')
# Display the plot
show(p)


# Customizing your scatter plots
# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color='blue', size=10, alpha=0.8)
# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color='red', size=10, alpha=0.8)
# Specify the name of the file
output_file('fert_lit_separate_colors.html')
# Display the plot
show(p)



# Lines
aapl = pd.read_csv('aapl.csv', index_col = 0, parse_dates = True)
date = pd.to_datetime(aapl['date'])
price = aapl['close']
# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime',x_axis_label='Date', y_axis_label='US Dollars')
# Plot date along the x axis and price along the y axis
p.line(date, price)
# Specify the name of the output file and show the result
output_file('line.html')
show(p)



# Lines and markers
# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')
# Plot date along the x-axis and price along the y-axis
p.line(date, price)
# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)
# Specify the name of the output file and show the result
output_file('line.html')
show(p)



# Patches
# NICHT LAUFFÄHIG (x und y sind Listen von Listen)
# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]
# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]
# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color='white')
# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)



# Plotting data from NumPy arrays
# Create array using np.linspace: x
x = np.linspace(0,5,100)
# Create array using np.cos: y
y = np.cos(x)
# Add circles at x and y
p = figure()
p.circle(x,y)
# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)



# Plotting data from Pandas DataFrames
# Read in the CSV file: df
df = pd.read_csv('auto-mpg.csv')
# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')
# Plot mpg vs hp by color
p.circle(df['hp'], df['mpg'], color=df['color'], size=10)
# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)



# The Bokeh ColumnDataSource (continued)
df = pd.read_csv('sprint.csv')
# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)
# Add circle glyphs to the figure p 
# p = figure(x_axis_label='Year', y_axis_label='Time')
p = figure(x_axis_label='Year', y_axis_label='Time')
p.circle('Year', 'Time', size=8, source=source, color='color')
# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)



# Selection and non-selection glyphs
# Create a figure with the "box_select" tool: p
p = figure(x_axis_label = 'Year', y_axis_label = 'Time', tools = 'box_select')
# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', selection_color = 'red', nonselection_alpha = 0.1, source = source)
# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)


# Hover glyphs
glucose = pd.read_csv('glucose.csv', index_col = 'datetime', parse_dates = True)
y = glucose['glucose']
x = range(0,len(y))
# Add circle glyphs to figure p
p = figure(x_axis_label = 'Time of day', y_axis_label = 'Blood glucose')
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')
# Add the hover tool to the figure p
p.add_tools(hover)
# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)



# Colormapping
df = pd.read_csv('auto-mpg.csv')
# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)
# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])
# Add a circle glyph to the figure p
p = figure(x_axis_label = 'weight', y_axis_label = 'mpg')
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')
# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)
