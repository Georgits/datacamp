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
# path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\12_Introduction_to_Data_Visualization_with_Python'
path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\12_Introduction_to_Data_Visualization_with_Python'
os.chdir(path)

# Chapter 1:  Customizing plots
data = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
year = data['Year']
physical_sciences = data['Physical Sciences']
computer_science = data['Computer Science']
health = data['Health Professions']
education = data['Education']

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color = 'blue')
# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color = 'red')
# Display the plot
plt.show()


# Using axes()
# Create plot axes for the first line plot
plt.axes([0.05,0.05,0.425, 0.9])
# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
# Create plot axes for the second line plot
plt.axes([0.525,0.05,0.425, 0.9])
# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
# Display the plot
plt.show()



# Using subplot() (1)
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1, 2, 1)
# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1, 2, 2)
# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')
# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()



# Using subplot() (2)
# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2,2,1)
# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2,2,2)
# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')
# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2,2,3)
# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')
# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2,2,4)
# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')
# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()



# Using xlim(), ylim()
# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')
# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')
# Set the x-axis range
plt.xlim(1990, 2010)
# Set the y-axis range
plt.ylim(0,50)
# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()
# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')



# Using axis()
# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')
# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')
# Set the x-axis and y-axis limits
plt.axis((1990,2010,0,50))
# Show the figure
plt.show()
# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')




# Using legend()
# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 
# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
# Add a legend at the lower center
plt.legend(loc='lower center')
# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()




# Using annotate()
# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')
# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()
# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]
# Add a black arrow annotation
plt.annotate('Maximum', xy = (yr_max, cs_max), xytext = (yr_max + 5, cs_max + 5), arrowprops=dict(facecolor='black'))
# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()



# Modifying styles
# Set the style to 'ggplot'
plt.style.use('ggplot')
# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 
# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')
# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))
# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')
# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')
# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()














# Chapter 2: Plotting 2D arrays
# Generating meshes
# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1,1,21)
# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u,v,)
# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 
# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()
# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


# Array orientation
# A = np.array([[1, 2, 1], [0, 0, 1], [-1, 1, 1]])
# A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
# A = np.array([[-1, 0, 1], [1, 0, 2], [1, 1, 1]])
# A = np.array([[1, 1, 1], [2, 0, 1], [1, 0, -1]])
A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()


# Contour & filled contour plots
# Generate a default contour map of the array Z
plt.subplot(2,2,1)
plt.contour(X, Y, Z)
# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)
# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)
# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)
# Improve the spacing between subplots
plt.tight_layout()
# Display the figure
plt.show()


# Modifying colormaps
# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')
# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')
# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')
# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')
# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


# Using hist2d()
auto = pd.read_csv('auto-mpg.csv')
hp = auto['hp']
mpg = auto['mpg']
# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins = (20,20), range = ((40,235),(8,48)))
# Add a color bar to the histogram
plt.colorbar()
# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()



# Using hexbin()
# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize = (15,12), extent = (40,235,8,48))
# Add a color bar to the histogram
plt.colorbar()
# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()


# Loading, examining images
# Load the image into an array: img
img = plt.imread('Astronaut-EVA.jpg')
# Print the shape of the image
print(img.shape)
# Display the image
plt.imshow(img)
# Hide the axes
plt.axis('off')
plt.show()



# Pseudocolor plot from image data
# Load the image into an array: img
img = plt.imread('Astronaut-EVA.jpg')
# Print the shape of the image
print(img.shape)
# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis = 2)
# Print the shape of the intensity
print(intensity.shape)
# Display the intensity with a colormap of 'gray'
plt.imshow(intensity, cmap = 'gray')
# Add a colorbar
plt.colorbar()
# Hide the axes and show the figure
plt.axis('off')
plt.show()



# Extent and aspect
# Load the image into an array: img
img = plt.imread('Astronaut-EVA.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)

# Improve spacing and display the figure
plt.tight_layout()
plt.show()


# Rescaling pixel intensities
# Load the image into an array: image
image = plt.imread('Unequalized_Hawkes_Bay_NZ.jpg')
# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))
# Rescale the pixels: rescaled_image
rescaled_image = 255*(image - pmin) / (pmax - pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." % 
      (rescaled_image.min(), rescaled_image.max()))
# Display the original image in the top subplot
plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)
# Display the rescaled image in the bottom subplot
plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow((rescaled_image*255).astype(np.uint8))
plt.show()
# Läuft nicht 100% so wie es laufen müsste
# https://stackoverflow.com/questions/49643907/clipping-input-data-to-the-valid-range-for-imshow-with-rgb-data-0-1-for-floa

















# Chapter 3: Statistical plots with Seaborn
# Simple linear regressions
auto = pd.read_csv('auto-mpg.csv')
# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x = 'weight', y = 'hp', data=auto)
# Display the plot
plt.show()

# Plotting residuals of a regression
# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')
# Display the plot
plt.show()

# Higher-order regressions
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, color='blue', scatter=None, label='order 1')
# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, color='green', scatter=None, label='order 2', order = 2)
# Add a legend and display the plot
plt.legend(loc = 'upper right')
plt.show()


# Grouping linear regressions by hue
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x = 'weight', y = 'hp', data = auto, hue = 'origin', palette = 'Set1')
# Display the plot
plt.style.use('ggplot')
plt.show()

# Grouping linear regressions by row or column
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x = 'weight', y = 'hp', data = auto, row = 'origin', palette = 'Set1')
# Display the plot
plt.show()


# Constructing strip plots
# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto)
# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, jitter = True, size = 3)
# Display the plot
plt.show()


# Constructing swarm plots
# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x='cyl', y='hp', data=auto)
# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x='hp', y='cyl', data=auto, hue='origin', orient = 'h')
# Display the plot
plt.show()


# Constructing violin plots
# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)
# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, color = 'lightgray', inner = None)
# Overlay a strip plot on the violin plot
sns.stripplot(x='cyl', y='hp', data=auto, jitter = True, size = 1.5)
# Display the plot
plt.show()


# Plotting joint distributions (1)
# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x='hp', y='mpg', data = auto)
# Display the plot
plt.show()


# Plotting joint distributions (2)
# The seaborn function sns.jointplot() has a parameter kind to specify how to visualize 
# the joint variation of two continuous random variables (i.e., two columns of a DataFrame)

#    kind='scatter' uses a scatter plot of the data points
#    kind='reg' uses a regression plot (default order 1)
#    kind='resid' uses a residual plot
#    kind='kde' uses a kernel density estimate of the joint distribution
#    kind='hex' uses a hexbin plot of the joint distribution
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x='hp', y='mpg', data=auto, kind = 'hex')
# Display the plot
plt.show()


# Plotting distributions pairwise (1)
# Print the first 5 rows of the DataFrame
print(auto.head())
# Plot the pairwise joint distributions from the DataFrame 
sns.pairplot(auto)
# Display the plot
plt.show()


# Plotting distributions pairwise (2)
# You will display regressions as well as scatter plots in the off-diagonal subplots. You will do this with the argument kind='reg' (where 'reg' means 'regression'). Another option for kind is 'scatter' (the default) that plots scatter plots in the off-diagonal subplots.
# You will also visualize the joint distributions separated by continent of origin. You will do this with the keyword argument hue specifying the 'origin'.
# Print the first 5 rows of the DataFrame
print(auto.head())
# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto, hue = 'origin', kind = 'reg')
# Display the plot
plt.show()
# Es werden keine Werte gezeichnet


# Visualizing correlations with a heatmap
cov_matrix = auto.corr()
# Print the covariance matrix
print(cov_matrix)
# Visualize the covariance matrix using a heatmap
sns.heatmap(cov_matrix)
# Display the heatmap
plt.show()



