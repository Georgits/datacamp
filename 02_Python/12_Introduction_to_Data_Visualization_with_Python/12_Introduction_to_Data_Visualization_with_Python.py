# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:22:49 2017

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\12_Introduction_to_Data_Visualization_with_Python'
path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\12_Introduction_to_Data_Visualization_with_Python'
os.chdir(path)

# Chapter 1:  Customizing plots
data = pd.read_csv('percent-bachelors-degrees-women-usa.csv')
year = data['Year']
physical_sciences = data['Physical Sciences']
computer_science = data['Computer Science']
# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color = 'blue')
# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color = 'red')
# Display the plot
plt.show()

