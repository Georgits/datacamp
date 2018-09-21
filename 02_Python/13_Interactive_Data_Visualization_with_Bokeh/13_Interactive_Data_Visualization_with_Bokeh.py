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



literacy_birth_rate = pd.read_csv('literacy_birth_rate.csv')
literacy_birth_rate = literacy_birth_rate.iloc[:162,:]
fertility = pd.to_numeric(literacy_birth_rate['fertility'])
female_literacy = pd.to_numeric(literacy_birth_rate['female_literacy'])
population = pd.to_numeric(literacy_birth_rate['population'])

literacy_birth_rate['fertility'] = pd.to_numeric(literacy_birth_rate['fertility'])
literacy_birth_rate['female_literacy'] = pd.to_numeric(literacy_birth_rate['female_literacy'])
literacy_birth_rate['population'] = pd.to_numeric(literacy_birth_rate['population'])

fertility_latinamerica = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'LAT']['fertility'])
female_literacy_latinamerica = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'LAT']['female_literacy'])

fertility_africa = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'AF']['fertility'])
female_literacy_africa = pd.to_numeric(literacy_birth_rate[literacy_birth_rate['Continent'] == 'AF']['female_literacy'])


latin_america = literacy_birth_rate[literacy_birth_rate['Continent'] == 'LAT']
africa = literacy_birth_rate[literacy_birth_rate['Continent'] == 'AF']


# Chapter 1:  Basic plotting with Bokeh
# A simple scatter plot
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



















# Chapter 2: Layouts, Interactions, and Annotations
# Creating rows of plots
# Convert df to a ColumnDataSource: source
source = ColumnDataSource(literacy_birth_rate)
# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)
# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source=source)
# Put p1 and p2 into a horizontal row: layout
layout = row(p1,p2)
# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)




# Creating columns of plots
# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')
# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)
# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')
# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source=source)
# Put plots p1 and p2 in a column: layout
layout = column(p1,p2)
# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)



# Nesting rows and columns of plots
df = pd.read_csv('auto-mpg.csv')
avg_mpg = df.groupby(['yr'])['mpg'].mean()
avg_mpg = pd.DataFrame(avg_mpg)
# avg_mpg.columns = ['avg_mpg']
source = ColumnDataSource(df)
source_avg = ColumnDataSource(avg_mpg)


mpg_hp = figure(x_axis_label='hp', y_axis_label='mpg')
mpg_hp.circle('hp', 'mpg', source = source)


mpg_weight = figure(x_axis_label='weight', y_axis_label='mpg')
mpg_weight.circle('weight', 'mpg', source = source)

avg_mpg = figure(x_axis_label='Year', y_axis_label='avg_mpg')
avg_mpg.line('yr', 'mpg', source = source_avg)


# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')
# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')
# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)



# Creating gridded layouts
# Create a list containing plots p1 and p2: row1
row1 = [p1,p2]
# Create a list containing plots p3 and p4: row2
row2 = [mpg_hp,mpg_weight]
# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])
# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)



# Starting tabbed layouts
# Create tab1 from plot p1: tab1
tab1 = Panel(child=mpg_hp, title='Latin America')
# Create tab2 from plot p2: tab2
tab2 = Panel(child=mpg_weight, title='Africa')
# Create tab3 from plot p3: tab3
tab3 = Panel(child=avg_mpg, title='Asia')
# Create tab4 from plot p4: tab4
tab4 = Panel(child=mpg_hp, title='Europe')



# Displaying tabbed layouts
### LÄUFT NICHT
# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])
# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)


# Linked axes
### LÄUFT NICHT
# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range
# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range
# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range
# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range
# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)



# Linked brushing
# Create ColumnDataSource: source
source = ColumnDataSource(literacy_birth_rate)
# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)', tools='box_select,lasso_select')
# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source=source)
# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)', tools='box_select,lasso_select')
# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)
# Create row layout of figures p1 and p2: layout
layout = row(p1,p2)
# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)



# How to create legends
latin_america = ColumnDataSource(latin_america)
africa = ColumnDataSource(africa)
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')
# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')
# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')
# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)


#Positioning and styling legends
# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'
# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'
# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)



# Adding a hover tooltip
# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country', '@Country')])
# Add the HoverTool object to figure p
p.add_tools(hover)
# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)



















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