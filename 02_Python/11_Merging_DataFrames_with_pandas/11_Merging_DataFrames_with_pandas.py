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



# Sorting DataFrame with the Index & columns
# NICHT LAUFFÄHIG !!!
# Read 'monthly_max_temp.csv' into a DataFrame: weather1
weather1 = pd.read_csv('monthly_max_temp.csv', index_col = 'Month')
# Print the head of weather1
print(weather1.head())
# Sort the index of weather1 in alphabetical order: weather2
weather2 = weather1.sort_index()
# Print the head of weather2
print(weather2.head())
# Sort the index of weather1 in reverse alphabetical order: weather3
weather3 = weather1.sort_index(ascending = False)
# Print the head of weather3
print(weather3.head())
# Sort weather1 numerically using the values of 'Max TemperatureF': weather4
weather4 = weather1.sort_values('Max TemperatureF')
# Print the head of weather4
print(weather4.head())


# Reindexing DataFrame from a list
# NICHT LAUFFÄHIG !!!
year = ['Jan',
 'Feb',
 'Mar',
 'Apr',
 'May',
 'Jun',
 'Jul',
 'Aug',
 'Sep',
 'Oct',
 'Nov',
 'Dec']
# Reindex weather1 using the list year: weather2
weather2 = weather1.reindex(year)
# Print weather2
print(weather2)
# Reindex weather1 using the list year with forward-fill: weather3
weather3 = weather1.reindex(year).ffill()
# Print weather3
print(weather3)



# Reindexing using another DataFrame Index
names_1981 = pd.read_csv('names1981.csv', header=None, names=['name','gender','count'], index_col=(0,1))
names_1881 = pd.read_csv('names1881.csv', header=None, names=['name','gender','count'], index_col=(0,1))
# Reindex names_1981 with index of names_1881: common_names
common_names = names_1981.reindex(names_1881.index)
# Print shape of common_names
print(common_names.shape)
# Drop rows with null counts: common_names
common_names = common_names.dropna()
# Print shape of new common_names
print(common_names.shape)



# Broadcasting in arithmetic formulas
weather = pd.read_csv('pittsburgh2013.csv', index_col = 'Date')
# Extract selected columns from weather as new DataFrame: temps_f
temps_f = weather[['Min TemperatureF', 'Mean TemperatureF', 'Max TemperatureF']]
# Convert temps_f to celsius: temps_c
temps_c = (temps_f - 32) * 5/9
# Rename 'F' in column names with 'C': temps_c.columns
temps_c.columns = temps_c.columns.str.replace('F','C')
# Print first 5 rows of temps_c
print(temps_c.head())



# Computing percentage growth of GDP
# Read 'GDP.csv' into a DataFrame: gdp
gdp = pd.read_csv('GDP_usa.csv', parse_dates=True, index_col='DATE')
# Slice all the gdp data from 2008 onward: post2008
post2008 = gdp.loc['2008':,]
# Print the last 8 rows of post2008
print(post2008.tail(8))
# Resample post2008 by year, keeping last(): yearly
yearly = post2008.resample('A').last()
# Print yearly
print(yearly)
# Compute percentage growth of yearly: yearly['growth']
yearly['growth'] = yearly.pct_change()*100
# Print yearly again
print(yearly)



# Converting currency of stocks
# Read 'sp500.csv' into a DataFrame: sp500
sp500 = pd.read_csv('sp500.csv', parse_dates=True, index_col='Date')
# Read 'exchange.csv' into a DataFrame: exchange
exchange = pd.read_csv('exchange.csv', parse_dates=True, index_col='Date')
# Subset 'Open' & 'Close' columns from sp500: dollars
dollars = sp500[['Open', 'Close']]
# Print the head of dollars
print(dollars.head(5))
# Convert dollars to pounds: pounds
pounds = dollars.multiply(exchange['GBP/USD'], axis = 'rows')
# Print the head of pounds
print(pounds.head())










# Chapter 2: Concatenating data
# Appending pandas Series
# Load 'sales-jan-2015.csv' into a DataFrame: jan
jan = pd.read_csv('sales-jan-2015.csv', parse_dates = True, index_col = 'Date')
# Load 'sales-feb-2015.csv' into a DataFrame: feb
feb = pd.read_csv('sales-feb-2015.csv', parse_dates = True, index_col = 'Date')
# Load 'sales-mar-2015.csv' into a DataFrame: mar
mar = pd.read_csv('sales-mar-2015.csv', parse_dates = True, index_col = 'Date')
# Extract the 'Units' column from jan: jan_units
jan_units = jan['Units']
# Extract the 'Units' column from feb: feb_units
feb_units = feb['Units']
# Extract the 'Units' column from mar: mar_units
mar_units = mar['Units']
# Append feb_units and then mar_units to jan_units: quarter1
quarter1 = jan_units.append(feb_units).append(mar_units)
# Print the first slice from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
# Print the second slice from quarter1
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])
# Compute & print total sales in quarter1
print(quarter1.sum())



# Concatenating pandas Series along row axis
# Initialize empty list: units
units = []
# Build the list of Series
for month in [jan, feb, mar]:
    print(month['Units'])
    units.append(month['Units'])
# Concatenate the list: quarter1
quarter1 = pd.concat(units, axis = 'rows')
# Print slices from quarter1
print(quarter1.loc['jan 27, 2015':'feb 2, 2015'])
print(quarter1.loc['feb 26, 2015':'mar 7, 2015'])



# Appending DataFrames with ignore_index
names_1881 =pd.read_csv('names1881.csv', index_col = 0 )
names_1981 = pd.read_csv('names1981.csv', index_col = 0 )
names_1881['year'] = 1881
names_1981['year'] = 1981
# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981, ignore_index = True)
# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)


pd.read_csv('names_1881.csv', index_col = 0)
names_1881['year'] = 1881
names_1981['year'] = 1981
# Append names_1981 after names_1881 with ignore_index=True: combined_names
combined_names = names_1881.append(names_1981, ignore_index = True)
# Print shapes of names_1981, names_1881, and combined_names
print(names_1981.shape)
print(names_1881.shape)
print(combined_names.shape)
# Print all rows that contain the name 'Morgan'
print(combined_names.loc[combined_names['name'] =='Morgan'])



# Concatenating pandas DataFrames along column axis
### NICHT LAUFFÄHIG !!!!
# Concatenate weather_max and weather_mean horizontally: weather
weather = pd.concat([weather_max, weather_mean], axis = 1)
# Print weather
print(weather)



# Reading multiple files to build a DataFrame
medals = []
medal_types = ['bronze', 'silver', 'gold']

for medal in medal_types:
    # Create the file name: file_name
    file_name = "%s_top5.csv" % medal
    # Create list of column names: columns
    columns = ['Country', medal]
    # Read file_name into a DataFrame: df
    medal_df = pd.read_csv(file_name, header = 0, index_col = 'Country', names=columns)
    # Append medal_df to medals
    medals.append(medal_df)
# Concatenate medals horizontally: medals
medals = pd.concat(medals, axis = 'columns')
# Print medals
print(medals)



# Concatenating vertically to get MultiIndexed rows
for medal in medal_types:
    file_name = "%s_top5.csv" % medal
    # Read file_name into a DataFrame: medal_df
    medal_df = pd.read_csv(file_name, header = 0, index_col = 'Country')
    # Append medal_df to medals
    medals.append(medal_df)
# Concatenate medals: medals
medals = pd.concat(medals, keys = ['bronze', 'silver', 'gold'], axis = 0)
# Print medals in entirety
print(medals)



# Slicing MultiIndexed DataFrames
# Sort the entries of medals: medals_sorted
medals_sorted = medals.sort_index(level =0)
# Print the number of Bronze medals won by Germany
print(medals_sorted.loc[('bronze','Germany')])
# Print data about silver medals
print(medals_sorted.loc['silver'])
# Create alias for pd.IndexSlice: idx
idx = pd.IndexSlice
# Print all the data on medals won by the United Kingdom
print(medals_sorted.loc[idx[:,'United Kingdom'],:])



# Concatenating horizontally to get MultiIndexed columns
hardware = pd.read_csv('feb-sales-Hardware.csv', index_col = 'Date')
software = pd.read_csv('feb-sales-Software.csv', index_col = 'Date')
service = pd.read_csv('feb-sales-Service.csv', index_col = 'Date')
dataframes = [hardware, software, service]
# Concatenate dataframes: february
february = pd.concat(dataframes, keys = ['Hardware', 'Software', 'Service'], axis = 1)
# Print february.info()
print(february.info())
# Assign pd.IndexSlice: idx
idx = pd.IndexSlice
# Create the slice: slice_2_8
slice_2_8 = february.loc['2015-02-02':'2015-02-08', idx[:, 'Company']]
# Print slice_2_8
print(slice_2_8)



# Concatenating DataFrames from a dict
jan = pd.read_csv('sales-jan-2015.csv', index_col = 'Date')
feb = pd.read_csv('sales-feb-2015.csv', index_col = 'Date')
mar = pd.read_csv('sales-mar-2015.csv', index_col = 'Date')
# Make the list of tuples: month_list
month_list = [('january', jan), ('february', feb), ('march', mar)]
# Create an empty dictionary: month_dict
month_dict = {}
for month_name, month_data in month_list:
    # Group month_data: month_dict[month_name]
    month_dict[month_name] = month_data.groupby('Company').sum() 
# Concatenate data in month_dict: sales
sales = pd.concat(month_dict)
# Print sales
print(sales)
# Print all sales by Mediacore
idx = pd.IndexSlice
print(sales.loc[idx[:, 'Mediacore'], :])



# Concatenating DataFrames with inner join
bronze = pd.read_csv('bronze_top5.csv', index_col = 'Country')
silver = pd.read_csv('silver_top5.csv', index_col = 'Country')
gold = pd.read_csv('gold_top5.csv', index_col = 'Country')
# Create the list of DataFrames: medal_list
medal_list = [bronze, silver, gold]
# Concatenate medal_list horizontally using an inner join: medals
medals = pd.concat(medal_list, keys = ['bronze', 'silver', 'gold'], axis = 1, join = 'inner')
# Print medals
print(medals)



# Resampling & concatenating DataFrames with inner join
china = pd.read_csv('gdp_china.csv', parse_dates=True, index_col = 'Year')
us = pd.read_csv('GDP_usa.csv', parse_dates=True, index_col='DATE')
# NICHT LAUFFÄHIG !!!!
# Resample and tidy china: china_annual
china_annual = china.resample('A').pct_change(10).dropna()
# Resample and tidy us: us_annual
us_annual = us.resample('A').pct_change(10).dropna()
# Concatenate china_annual and us_annual: gdp
gdp = pd.concat([china_annual, us_annual], axis = 1, join = 'inner')
# Resample gdp and print
print(gdp.resample('10A').last())