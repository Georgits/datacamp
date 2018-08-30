# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:22:49 2017

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\09_Manipulating_DataFrames_with_pandas'
os.chdir(path)



# Chapter 1: Extracting and transforming data


# Index ordering
# In this exercise, the DataFrame election is provided for you. 
# It contains the 2012 US election results for the state of Pennsylvania with 
# county names as row indices. Your job is to select 'Bedford' county and the'winner' 
# column. Which method is the preferred way?
election = pd.read_csv('pennsylvania2012_turnout.csv', index_col='county')
election.loc['Bedford', 'winner']
election['winner']['Bedford']



# Positional and labeled indexing
election.head()
# Assign the row position of election.loc['Bedford']: x
x = 4
# Assign the column position of election['winner']: y
y = 4
# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])


# Indexing and column rearrangement
filename = 'https://s3.amazonaws.com/assets.datacamp.com/production/course_1650/datasets/pennsylvania2012.csv'
# Read in filename and set the index: election
election = pd.read_csv(filename, index_col='county')
# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]
# Print the output of results.head()
print(results.head())


# Slicing rows
# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry':'Potter',:]
# Print the p_counties DataFrame
print(p_counties)
# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1]
# Print the p_counties_rev DataFrame
print(p_counties_rev)


# Slicing columns
# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']
# Print the output of left_columns.head()
print(left_columns.head())
# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']
# Print the output of middle_columns.head()
print(middle_columns.head())
# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]
# Print the output of right_columns.head()
print(right_columns.head())


# Subselecting DataFrames with lists
# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']
# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']
# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]
# Print the three_counties DataFrame
print(three_counties)


# Thresholding data
election = pd.read_csv('pennsylvania2012_turnout.csv', index_col='county')
# Create the boolean array: high_turnout
high_turnout = election['turnout'] > 70
# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]
# Print the high_turnout_results DataFrame
print(high_turnout_df)



# Filtering columns using other columns
# Import numpy
import numpy as np
# Create the boolean array: too_close
too_close = election['margin'] < 1
# Assign np.nan to the 'winner' column where the results were too close to call
election['winner'][too_close] = np.nan
# Print the output of election.info()
print(election.info())


# Filtering using NaNs
# Select the 'age' and 'cabin' columns: df
df = titanic[['age','cabin']]
# Print the shape of df
print(df.shape)
# Drop rows in df with how='any' and print the shape
print(df.dropna(how = 'any').shape)
# Drop rows in df with how='all' and print the shape
print(df.dropna(how = 'all').shape)
# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())




# Using apply() to transform a column
weather = pd.read_csv('pittsburgh2013.csv', index_col='Date')

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)
# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)
# Reassign the columns df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']
# Print the output of df_celsius.head()
print(df_celsius.head())



# Using .map() with a dictionary
# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}
# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)
# Print the output of election.head()
print(election.head())


# Using vectorized functions
# Import zscore from scipy.stats
from scipy.stats import zscore
# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])
# Print the type of turnout_zscore
print(type(turnout_zscore))
# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore
# Print the output of election.head()
print(election.head())







# Chapter 2: Advanced indexing

# Index values and names
sales = pd.read_csv('sales.csv', index_col='month')
print(sales)
sales.index = range(len(sales))
print(sales)


# Changing index of a DataFrame
# Create the list of new indexes: new_idx
new_idx = [i.upper() for i in sales.index]
# Assign new_idx to sales.index
sales.index = new_idx
# Print the sales DataFrame
print(sales)


# Changing index name labels
# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTH'
# Print the sales DataFrame
print(sales)
# Assign the string 'PRODUCTS' to sales.columns.name 
sales.columns.name = 'PRODUCTS'
# Print the sales dataframe again
print(sales)



# Building an index, then a DataFrame
# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
# Assign months to sales.index
sales.index = months
# Print the modified sales DataFrame
print(sales)


# Extracting data with a MultiIndex
sales['state'] = ['CA','CA','NY','NY','TX','TX',]
sales['month'] = [1,2,1,2,1,2]
# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])
# Print sales.loc[['CA', 'TX']]
print(sales.loc[['CA', 'TX']])
# Print sales['CA':'TX']
print(sales['CA':'TX'])



# Setting & sorting a MultiIndex
# Sort the MultiIndex: sales
sales = sales.sort_index()
# Print the sales DataFrame
print(sales)



# Using .loc[] with nonunique indexes
sales = pd.read_csv('sales.csv', index_col='month')
sales['state'] = ['CA','CA','NY','NY','TX','TX',]
sales['month'] = [1,2,1,2,1,2]
# Set the index to the column 'state': sales
sales = sales.set_index(['state'])
# Print the sales DataFrame
print(sales)
# Access the data from 'NY'
print(sales.loc['NY'])




# Indexing multiple levels of a MultiIndex
sales = pd.read_csv('sales.csv', index_col='month')
sales['state'] = ['CA','CA','NY','NY','TX','TX',]
sales['month'] = [1,2,1,2,1,2]
sales = sales.set_index(['state', 'month'])
sales = sales.sort_index()
# Look up data for NY in month 1: NY_month1
NY_month1 = sales.loc[('NY',1)]
# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA','TX'],2),:]
# Look up data for all states in month 2: all_month2
all_month2 = sales.loc[(slice(None),2),:]








# Chapter 3: Rearranging and reshaping data
users = pd.read_csv('users.csv', index_col=0)
users
# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index = 'weekday', columns = 'city', values = 'visitors')
# Print the pivoted DataFrame
print(visitors_pivot)



# Pivoting all variables
# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index = 'weekday', columns = 'city', values = 'signups')
# Print signups_pivot
print(signups_pivot)
# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index = 'weekday', columns = 'city')
# Print the pivoted DataFrame
print(pivot)



# Stacking & unstacking I
users = pd.read_csv('users.csv', index_col=0)
users = users.set_index(['city', 'weekday'])
users = users.sort_index()
# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level = 'weekday')
# Print the byweekday DataFrame
print(byweekday)
# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level = 'weekday'))



# Stacking & unstacking II
# Unstack users by 'city': bycity
bycity = users.unstack(level = 'city')
# Print the bycity DataFrame
print(bycity)
# Stack bycity by 'city' and print it
print(bycity.stack(level = 'city'))



# Restoring the index order
# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level = 'city')
# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0,1)
# Print newusers and verify that the index is not sorted
print(newusers)
# Sort the index of newusers: newusers
newusers = newusers.sort_index()
# Print newusers and verify that the index is now sorted
print(newusers)
# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))



# Adding names for readability
visitors_by_city_weekday = signups_pivot
# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index() 
# Print visitors_by_city_weekday
print(visitors_by_city_weekday)
# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')
# Print visitors
print(visitors)



# Going from wide to long
users = pd.read_csv('users.csv', index_col=0)
# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday', 'city'])
# Print skinny
print(skinny)



# Obtaining key-value pairs with melt()
# Set the new index: users_idx
users_idx = users.set_index(['city','weekday'])
# Print the users_idx DataFrame
print(users_idx)
# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)
# Print the key-value pairs
print(kv_pairs)
