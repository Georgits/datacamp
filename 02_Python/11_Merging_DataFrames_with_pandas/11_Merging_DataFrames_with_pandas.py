# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:22:49 2017

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\10_Introduction_to_Databases_in_Python'
path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\11_Merging_DataFrames_with_pandas'
os.chdir(path)


# Chapter 1: Preparing data
# Reading DataFrames from multiple files
# Read 'Bronze.csv' into a DataFrame: bronze
bronze = pd.read_csv('Bronze.csv')
# Read 'Silver.csv' into a DataFrame: silver
silver = pd.read_csv('Silver.csv')
# Read 'Gold.csv' into a DataFrame: gold
gold = pd.read_csv('Gold.csv')
# Print the first five rows of gold
print(gold.head())


# Reading DataFrames from multiple files in a loop
# Create the list of file names: filenames
filenames = ['Gold.csv', 'Silver.csv', 'Bronze.csv']
# Create the list of three DataFrames: dataframes
dataframes = []
for filename in filenames:
    dataframes.append(pd.read_csv(filename))
# Print top 5 rows of 1st DataFrame in dataframes
print(dataframes[0].head())



#Combining DataFrames from multiple data files
# Make a copy of gold: medals
medals = gold.copy()
# Create list of new column labels: new_labels
new_labels = ['NOC', 'Country', 'Gold']
# Rename the columns of medals using new_labels
medals.columns = new_labels
# Add columns 'Silver' & 'Bronze' to medals
medals['Silver'] = silver['Total']
medals['Bronze'] = bronze['Total']
# Print the head of medals
print(medals.head())