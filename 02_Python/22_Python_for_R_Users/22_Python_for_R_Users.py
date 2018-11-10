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
