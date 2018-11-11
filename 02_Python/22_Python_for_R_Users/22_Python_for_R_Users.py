# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:52:34 2018

@author: georg
"""



# import inspect
# lines = inspect.getsource(foo)
# print(lines)


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
import matplotlib.pyplot as plt


plt.style.use('ggplot')
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\22_Python_for_R_Users'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\22_Python_for_R_Users'
os.chdir(path)


tips = pd.read_csv('tips.csv')


# Chapter 1: The Basics
# Assignment and data types
# Assign an integer (42) to w
w = 42
# Assign a float (3.14) to x
x = 3.14
# Assign a boolean (True) to y
y = True
# Assign a string ('python') to z
z = 'python'
# Print the data types of y and z
print(type(y))
print(type(z))



# Arithmetic with strings
# Add 'the quick' to 'brown fox'
print('the quick' + 'brown fox')
# Assign 'jump' to the variable x
x = 'jump'
# Multiply x by 3
print(x * 3)
# Have the string 'lazy' next to the string 'dog'
print('lazy' 'dog')



# Lists
# Assign the values to the list
person_list = ['Jonathan', 'Cornelissen', 'male', True, 458]
# Get the first name from the list
print(person_list[0])
# Get the first and last name from the list
print(person_list[0:2])
# Get the employment status
print(person_list[-2])


# Dictionaries
# Create a dictionary from of the employee information list
person_dict = {
    'fname': person_list[0],
    'lname': person_list[1],
    'sex': person_list[2],
    'employed': person_list[3],
    'twitter_followers': person_list[4]
}

# Get the first and last names from the dict
print(person_dict['fname'])
print(person_dict['lname'])




# Methods
# Append values to a list
person_list.append(2018)
# Print person_list
print(person_list)
# Print the last element of person_list
print(person_list[-1])


# Update the person_dict dictionary
person_dict = {'twitter_followers': 450, 'sex': 'male', 'fname': 'Jonathan', 'lname': 'Cornelissen', 'employed': True}
person_dict.update({'date': '2018-06', 'twitter_followers': 458})
# Print the person_dict dictionary
print(person_dict)



# NumPy arrays
# Import the numpy library with an alias: np
import numpy as np
# Load the boston dataset
boston = np.loadtxt('boston_data.csv', delimiter=',')
# Get the first row of data
first = boston[0]
# Calculate its mean
print(first.mean())



# Pandas DataFrames
# Import the pandas library
import pandas as pd
# Load the tips.csv dataset
tips = pd.read_csv('tips.csv')
# Look at the first 5 rows
print(tips.head())









# Chapter 2: Control flow, Loops, and Functions
# Control flow
# Assign 5 to a variable
num_drinks = 5

# if statement
if num_drinks < 0:
    print('error')
# elif statement
elif num_drinks <= 4:
    print('non-binge')
# else statement
else:
    print('binge')


# Loops
num_drinks = [5, 4, 3, 3, 3, 5, 6, 10]

# Write a for loop
for drink in num_drinks:
    # if/else statement
    if drink <= 4:
        print('non-binge')
    else:
        print('binge')    
        
        
# Individual binge drinking function
# Binge status for males
def binge_male(num_drinks):
    if num_drinks <= 5:
        return 'non-binge'
    else:
        return 'binge'
# Check
print(binge_male(6))


# Binge status for females
def binge_female(num_drinks):
    if num_drinks <= 4:
        return 'non-binge'
    else:
        return 'binge'
# Check
print(binge_female(2))




# General binge drinking function
# A function that returns a binge status
def binge_status(sex, num_drinks):
    if sex == 'male':
        return binge_male(num_drinks)
    else:
        return binge_female(num_drinks)
    
# Male who had 5 drinks
print(binge_status('male', 5))

# Female who had 5 drinks
print(binge_status('female', 5))



# Lambda functions
# A function that takes a value and returns its square
def sq_func(x):
    return(x**2)
    
# A lambda function that takes a value and returns its square
sq_lambda = lambda x: x**2

# Use the lambda function
print(sq_lambda(3))




# Mapping functions
# map the binge_male function to num_drinks
print(list(map(binge_male, num_drinks)))

# map the binge_female function to num_drinks
print(list(map(binge_female, num_drinks)))


# List comprehension
inflam_files =  ['inflammation-02.csv', 'inflammation-03.csv', 'inflammation-01.csv']

# Append dataframes into list with for loop
dfs_list = []
for f in inflam_files:
    dat = pd.read_csv(f)
    dfs_list.append(dat)

# Re-write the provided for loop as a list comprehension: dfs_comp
dfs_comp = [pd.read_csv(f) for f in inflam_files]
print(dfs_comp)



# Dictionary comprehension
twitter_followers = [['jonathan', 458], ['daniel', 660], ['hugo', 3509], ['datacamp', 26400]]
# Write a dict comprehension
tf_dict = {key:value for (key, value) in twitter_followers}
# Print tf_dict
print(tf_dict)









# Chapter 3: Pandas
# Selecting columns
# Print the tip column using dot notation
print(tips.tip)
# Print the sex column using square bracket notation
print(tips['sex'])
# Print the tip and sex columns
print(tips[['tip', 'sex']])

# Print the first row of tips using iloc
print(tips.iloc[0])
# Print all the rows where sex is Female
print(tips.loc[tips.sex == 'Female'])
# Print all the rows where sex is Female and total_bill is greater than 15
print(tips.loc[(tips.sex == 'Female') & (tips.total_bill > 15)])




# Selecting rows and columns
# Subset rows and columns
print(tips.loc[tips['sex'] == 'Female', ['total_bill', 'tip', 'sex']])

# 3 rows and 3 columns with iloc
print(tips.iloc[0:3, 0:3])



# Integers and floats
#Inspect the output of tips.dtypes in the shell
tips.dtypes
# Convert the size column
tips['size'] = tips['size'].astype(int)
# Convert the tip column
tips['tip'] = tips['tip'].astype(float)
    # Look at the types
print(tips.dtypes)



# Strings
# Inspect the 'sex' and 'smoker' columns in the shell
tips[['sex', 'smoker']]
# Convert sex to lower case
tips['sex'] = tips['sex'].str.lower()
# Convert smoker to upper case
tips['smoker'] = tips['smoker'].str.upper()
# Print the sex and smoker columns
print(tips[['sex', 'smoker']])


# Category
tips['time']
# Convert the type of time column
tips['time'] = tips['time'].astype('category')
# Use the cat accessor to print the categories in the time column
print(tips['time'].cat.categories)
# Order the time category so lunch is before dinner
tips['time2'] = tips['time'].cat.reorder_categories(['Lunch', 'Dinner'], ordered=True)
# Use the cat accessor to print the categories in the time2 column
print(tips['time2'].cat.categories)



# Dates (I)
# Load the country_timeseries dataset
ebola = pd.read_csv('country_timeseries.csv')
# Inspect the Date column
print(ebola['Date'].dtype)
ebola['Date']
# Convert the type of Date column into datetime
ebola['Date'] = pd.to_datetime(ebola['Date'], format='%m/%d/%Y')
# Inspect the Date column
print(ebola['Date'].dtype)
ebola['Date']



# Dates (II)
# Load the dataset and ensure Date column is imported as datetime
ebola = pd.read_csv('country_timeseries.csv', parse_dates=['Date'])
# Inspect the Date column
print(ebola['Date'].dtype)
# Create a year, month, day column using the dt accessor
ebola['year'] = ebola.Date.dt.year
ebola['month'] = ebola.Date.dt.month
ebola['day'] = ebola.Date.dt.day
# Inspect the newly created columns
print(ebola[['year', 'month', 'day']].head())



# Missing values
# Print the rows where total_bill is missing
print(tips.loc[pd.isnull(tips['total_bill'])])
# Mean of the total_bill column
tbill_mean = tips['total_bill'].mean()
# Fill in missing total_bill
print(tips['total_bill'].fillna(tbill_mean))



# Print the rows where total_bill is missing
print(tips.loc[pd.isnull(tips['total_bill'])])
# Mean of the total_bill column
tbill_mean = tips['total_bill'].mean()
# Fill in missing total_bill
print(tips['total_bill'].fillna(tbill_mean))
# You can also drop missing values using the .dropna() method



# Groupby
# Mean tip by sex
print(tips.groupby('sex')['tip'].mean())
# Mean tip by sex and time
print(tips.groupby(['sex', 'time'])['tip'].mean())
# In addition to calculating the mean, you can use other methods such as .agg() and .filter() on grouped DataFrames.



# Tidy data
airquality = pd.read_csv('airquality.csv')
# Melt the airquality DataFrame
airquality_melted = pd.melt(airquality, id_vars=['Day', 'Month'])
print(airquality_melted)
# Pivot the molten DataFrame
airquality_pivoted = airquality_melted.pivot_table(index=['Month', 'Day'], columns='variable', values='value')
print(airquality_pivoted)
# Reset the index
print(airquality_pivoted.reset_index())








# Chapter 4: Plotting
# Univariate plots in pandas
# Histogram of tip
tips['tip'].plot(kind = 'hist')
plt.show()

# Boxplot of the tip column
tips['tip'].plot(kind = 'box')
plt.show()

# Bar plot
cts = tips['sex'].value_counts()
cts.plot(kind = 'bar')
plt.show()


# Bivariate plots in pandas
# Scatter plot between the tip and total_bill
tips.plot(x = 'total_bill', y = 'tip', kind = 'scatter')
plt.show()

# Boxplot of the tip column by sex
tips.boxplot(column='tip', by='sex')
plt.show()



# Univariate plots in seaborn
# Bar plot
sns.countplot(x='sex', data=tips)
plt.show()

# Histogram
sns.distplot(tips['total_bill'])
plt.show()


# Bivariate plots in seaborn
# Boxplot for tip by sex
sns.boxplot(x='sex', y='tip', data=tips)
plt.show()

# Scatter plot of total_bill and tip
sns.regplot(x='total_bill', y='tip', data=tips)
plt.show()



# Facet plots in seaborn
# Scatter plot of total_bill and tip faceted by smoker and colored by sex
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', col='smoker')
plt.show()

# FacetGrid of time and smoker colored by sex
facet = sns.FacetGrid(tips, col='time', row='smoker', hue='sex')
# Map the scatter plot of total_bill and tip to the FacetGrid
facet.map(plt.scatter, 'total_bill', 'tip')
plt.show()



# Univariate and bivariate plots in matplotlib
# Univariate histogram
plt.hist(tips['total_bill'])
plt.show()

# Bivariate scatterplot
plt.scatter(tips['tip'], tips['total_bill'])
plt.show()


# Subfigures in matplotlib
# Create a figure with 1 axes
fig, ax = plt.subplots(1, 1)

# Plot a scatter plot in the axes
ax.scatter(tips['tip'], tips['total_bill'])
plt.show()


# Create a figure with scatter plot and histogram
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(tips['tip'], tips['total_bill'])
ax2.hist(tips['total_bill'])
plt.show()



# Working with axes
# Distplot of tip
dis = sns.distplot(tips['tip'])
# Print the type
print(type(dis))


# Figure with 2 axes: regplot and distplot
fig, (ax1, ax2) = plt.subplots(1,2)
sns.distplot(tips['tip'], ax=ax1)
sns.regplot(x='total_bill', y='tip', data=tips, ax=ax2)
plt.show()



# Polishing up a figure
# Create a figure with 1 axes
fig, ax = plt.subplots()
# Draw a displot
ax = sns.distplot(tips['total_bill'])
# Label the title and x axis
ax.set_title('Histogram')
ax.set_xlabel('Total Bill')
plt.show()



# Chapter 5: Capstone
# Load multiple data files
import glob
# Get a list of all the csv files
csv_files = glob.glob('*.csv')
# List comprehension that loads of all the files
dfs = [pd.read_csv(x) for x in csv_files]
# List comprehension that looks at the shape of all DataFrames
print([x.shape for x in dfs])
print(csv_files)


# Explore
# Get the planes DataFrame
planes = dfs[4]
# Count the frequency of engines in our data
print(planes['engines'].value_counts())
# Look at all planes with >= 3 engines
print(planes.loc[planes['engines'] >= 3])
# Look at all planes with >= 3 engines and < 100 seats
print(planes.loc[(planes['engines'] >= 3) & (planes['seats']<= 100)])



# Visualize
# Scatter plot of engines and seats
planes.plot(x='engines', y='seats', kind='scatter')
plt.show()

# Histogram of seats
planes['seats'].plot(kind = 'hist')
plt.show()

# Boxplot of seats by engine
planes.boxplot(column='seats', by='engine')
plt.xticks(rotation=45)
plt.show()



# Recode dates
# We defined the get_season() function that converts a given date to a season (one of winter, spring, summer, and fall)
def calculate_season(month, day):
    if month < 3:
        return('winter')
    elif month == 3:
        if day < 20:
            return('winter')
        else:
            return('spring')
    elif month < 6:
        return('spring')
    elif month == 6:
        if day < 21:
            return('spring')
        else:
            return('summer')
    elif month < 9:
        return('summer')
    elif month == 9:
        if day < 22:
            return('summer')
        else:
            return('fall')
    elif month < 12:
        return('fall')
    elif month == 12:
        if day < 21:
            return('fall')
        else:
            return('winter')
    else:
        return(np.NaN)

def get_season(time_hour):
    y_m_dt = time_hour.split('-')
    month = int(y_m_dt[1])
    
    d_t = y_m_dt[2].split(' ')
    day = int(d_t[0])
    return(calculate_season(month, day))


flights = dfs[2]
# Print time_hour
print(flights['time_hour'])
# Apply the function on data
flights['season'] = flights['time_hour'].apply(get_season)
# Print time_hour and season
print(flights[['time_hour', 'season']])



# Groupby aggregates
# Calculate total_delay
flights['total_delay'] = flights['dep_delay'] + flights['arr_delay']
# Mean total_delay by carrier
tdel_car = flights.groupby('carrier')['total_delay'].mean().reset_index()
print(tdel_car)
# Mean dep_delay and arr_delay for each season
dadel_season = flights.groupby('season')['dep_delay', 'arr_delay'].mean().reset_index()
print(dadel_season)
# Mean and std delays by origin
del_ori = flights.groupby('origin')['total_delay', 'dep_delay', 'arr_delay'].agg(['mean', 'std'])
print(del_ori)




# Plots
# Create a figure
fig, (ax1, ax2) = plt.subplots(2,1)
# Boxplot and barplot in the axes
sns.boxplot(x='origin', y='dep_delay', data=flights, ax=ax1)
sns.barplot(x='carrier', y='total_delay', data=tdel_car, ax=ax2)
# Label axes
ax1.set_title('Originating airport and the departure delay')
# Use tight_layout() so the plots don't overlap
fig.tight_layout()
plt.show()



# Dummy variables
# Look at the head of flights_sub
print(flights.head())
# Create dummy variables
flights_dummies = pd.get_dummies(flights)
# Look at the head of flights_dummies
print(flights_dummies.head())