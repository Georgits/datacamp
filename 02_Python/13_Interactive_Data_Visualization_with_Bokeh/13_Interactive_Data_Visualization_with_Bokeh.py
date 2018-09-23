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
from numpy.random import random
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
from bokeh.models import Slider, ColumnDataSource
# Import CheckboxGroup, RadioGroup, Toggle from bokeh.models
from bokeh.models import CheckboxGroup, RadioGroup, Toggle





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


# Multiple sliders in one document
# Create first slider: slider1
slider1 = Slider(title='slider1', start=0, end=10, step=0.1, value=2)
# Create second slider: slider2
slider2 = Slider(title='slider2', start=10, end=100, step=1, value=20)
# Add slider1 and slider2 to a widgetbox
layout = widgetbox(slider1,slider2)
# Add the layout to the current document
curdoc().add_root(layout)


# How to combine Bokeh models into layouts
N = 300
x = random(N)
y = random(N)
# Create ColumnDataSource: source
source = ColumnDataSource(data={'x': x, 'y': y})
# Add a line to the plot
plot.line('x', 'y', source=source)
# Create a column layout: layout
layout = column(widgetbox(slider), plot)
# Add the layout to the current document
curdoc().add_root(layout)


# Learn about widget callbacks
# Define a callback function: callback
def callback(attr, old, new):
    # Read the current value of the slider: scale
    scale = slider.value
    # Compute the updated y using np.sin(scale/x): new_y
    new_y = np.sin(scale/x)
    # Update source with the new data values
    source.data = {'x': x, 'y': new_y}
# Attach the callback to the 'value' property of slider
slider.on_change('value', callback)
# Create layout and add to current document
layout = column(widgetbox(slider), plot)
curdoc().add_root(layout)



# Updating data sources from dropdown callbacks
# Create ColumnDataSource: source
source = ColumnDataSource(data={
    'x' : fertility,
    'y' : female_literacy
})
# Create a new plot: plot
plot = figure()
# Add circles to the plot
plot.circle('x', 'y', source=source)
# Define a callback function: update_plot
def update_plot(attr, old, new):
    # If the new Selection is 'female_literacy', update 'y' to female_literacy
    if new == 'female_literacy': 
        source.data = {
            'x' : fertility,
            'y' : female_literacy
        }
    # Else, update 'y' to population
    else:
        source.data = {
            'x' : fertility,
            'y' : population
        }
# Create a dropdown Select widget: select    
select = Select(title="distribution", options=['female_literacy', 'population'], value='female_literacy')
# Attach the update_plot callback to the 'value' property of select
select.on_change('value', update_plot)
# Create layout and add to current document
layout = row(select, plot)
curdoc().add_root(layout)


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



# Button widgets
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
# Create layout and add to current document
layout = column(widgetbox(button), plot)
curdoc().add_root(layout)




# Button styles
# Add a Toggle: toggle
toggle = Toggle(button_type='success', label = 'Toggle button')
# Add a CheckboxGroup: checkbox
checkbox = CheckboxGroup(labels=['Option 1', 'Option 2', 'Option 3'])
# Add a RadioGroup: radio
radio = RadioGroup(labels=['Option 1', 'Option 2', 'Option 3'])
# Add widgetbox(toggle, checkbox, radio) to the current document
curdoc().add_root(widgetbox(toggle, checkbox, radio))














# Chapter 4: Putting It All Together! A Case Study
# Introducing the project dataset
data = pd.read_csv('gapminder_tidy.csv', index_col = 'Year')
data['fertility'] = pd.to_numeric(data['fertility'])
data['life'] = pd.to_numeric(data['life'])
data['population'] = pd.to_numeric(data['population'])
data['child_mortality'] = pd.to_numeric(data['child_mortality'])
data['gdp'] = pd.to_numeric(data['gdp'])

# Some exploratory plots of the data
# Perform necessary imports
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country' : data.loc[1970].Country,
})

# Create the figure: p
p = figure(title='1970', x_axis_label='Fertility (children per woman)', y_axis_label='Life Expectancy (years)',
           plot_height=400, plot_width=700,
           tools=[HoverTool(tooltips='@country')])

# Add a circle glyph to the figure p
p.circle(x='x', y='y', source=source)

# Output the file and show the figure
output_file('gapminder.html')
show(p)



# Beginning with just a plot
# Import the necessary modules
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    'pop'      : (data.loc[1970].population / 20000000) + 2,
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax))

# Add circle glyphs to the plot
plot.circle (x='x', y='y', fill_alpha=0.8, source=source)

# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

# Add the plot to the current document and add a title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'


# Enhancing the plot with some shading
# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()
# Import CategoricalColorMapper from bokeh.models and the Spectral6 palette from bokeh.palettes
from bokeh.models import CategoricalColorMapper
from bokeh.palettes import Spectral6
# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', fill_alpha=0.8, source=source,
            color=dict(field='region', transform=color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right'
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().add_root(plot)
curdoc().title = 'Gapminder'


# # Adding a slider to vary the year
# Import the necessary modules
from bokeh.layouts import widgetbox, row
from bokeh.models import Slider
# Define the callback function: update_plot
def update_plot(attr, old, new):
    # set the `yr` name to `slider.value` and `source.data = new_data`
    yr = slider.value
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    source.data  = new_data
# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')
# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)
# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)


# Customizing based on user input
# Define the callback function: update_plot
def update_plot(attr, old, new):
    # Assign the value of the slider: yr
    yr = slider.value
    # Set new_data
    new_data = {
        'x'       : data.loc[yr].fertility,
        'y'       : data.loc[yr].life,
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
    }
    # Assign new_data to: source.data
    source.data = new_data
    # Add title to figure: plot.title.text
    plot.title.text = 'Gapminder data for %d' % yr
# Make a slider object: slider
slider = Slider(start = 1970, end = 2010, step = 1, value = 1970, title = 'Year')

# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)

# Make a row layout of widgetbox(slider) and plot and add it to the current document
layout = row(widgetbox(slider), plot)
curdoc().add_root(layout)



# Adding a hover tool
# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool: hover
hover = HoverTool(tooltips=[('Country', '@country')])

# Add the HoverTool to the plot
plot.add_tools(hover)

# Create layout: layout
layout = row(widgetbox(slider), plot)

# Add layout to current document
curdoc().add_root(layout)





# Adding dropdowns to the app
# Define the callback: update_plot
def update_plot(attr, old, new):
    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    # Label axes of plot
    plot.xaxis.axis_label = x
    plot.yaxis.axis_label = y
    # Set new_data
    new_data = {
        'x'       : data.loc[yr][x],
        'y'       : data.loc[yr][y],
        'country' : data.loc[yr].Country,
        'pop'     : (data.loc[yr].population / 20000000) + 2,
        'region'  : data.loc[yr].region,
        'gdp'     : data.loc[yr].gdp,
    }
    # Assign new_data to source.data
    source.data = new_data

    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr
# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')
# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)
# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)
# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)
# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)
# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)
# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select), plot)
curdoc().add_root(layout)
