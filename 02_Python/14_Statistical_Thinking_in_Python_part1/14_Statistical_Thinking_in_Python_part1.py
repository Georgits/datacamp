# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:48:53 2018

@author: d91067
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\14_Statistical_Thinking_in_Python_part1'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\14_Statistical_Thinking_in_Python_part1'
os.chdir(path)

iris = pd.read_csv('iris.csv')
setosa_petal_length = pd.to_numeric(iris[iris['species'] == 'setosa']['petal_length'])
versicolor_petal_length = pd.to_numeric(iris[iris['species'] == 'versicolor']['petal_length'])
virginica_petal_length = pd.to_numeric(iris[iris['species'] == 'virginica']['petal_length'])

setosa_petal_width = pd.to_numeric(iris[iris['species'] == 'setosa']['petal_width'])
versicolor_petal_width = pd.to_numeric(iris[iris['species'] == 'versicolor']['petal_width'])
virginica_petal_width = pd.to_numeric(iris[iris['species'] == 'virginica']['petal_width'])


# Chapter 1: Graphical exploratory data analysis
# Plotting a histogram of iris data
# Set default Seaborn style
sns.set()
# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)
# Show histogram
plt.show()

# Axis labels!
# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)
# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')
# Show histogram
plt.show()

# Adjusting the number of bins in a histogram
# Compute number of data points: n_data
n_data = len(versicolor_petal_length)
# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)
# Convert number of bins to integer: n_bins
n_bins = int(n_bins)
# Plot the histogram
_ = plt.hist(versicolor_petal_length, bins = n_bins)
# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')
# Show histogram
plt.show()

# Bee swarm plot
# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x='species', y='petal_length', data=iris)
# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')
# Show the plot
plt.show()


# Computing the ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n
    return x, y


# Plotting the ECDF
# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)
# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
# Label the axes
_ = plt.xlabel('versicolor_petal_length')
_ = plt.ylabel('ECDF')
# Display the plot
plt.show()


# Comparison of ECDFs
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)
# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')
# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
# Display the plot
plt.show()








# Chapter 2: Quantitative exploratory data analysis
# Computing means
# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)
# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')

# Computing percentiles
# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])
# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)
# Print the result
print(ptiles_vers)


# Comparing percentiles to ECDF
# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')
# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')
# Show the plot
plt.show()


# Box-and-whisker plot
# Create box plot with Seaborn's default settings
_ = sns.boxplot(x='species', y='petal_length', data=iris)
# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')
# Show the plot
plt.show()


# Computing the variance
# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)
# Square the differences: diff_sq
diff_sq = differences **2
# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)
# Compute the variance using NumPy: variance_np = 
variance_np = np.var(versicolor_petal_length)
# Print the results
print(variance_explicit, variance_np)


# The standard deviation and the variance
# Compute the variance: variance
variance = np.var(versicolor_petal_length)
# Print the square root of the variance
print(np.sqrt(variance))
# Print the standard deviation
print(np.std(versicolor_petal_length)) 


# Scatter plots
# Make a scatter plot
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
# Label the axes
_ = plt.xlabel('versicolor: petal length (cm)')
_ = plt.ylabel('versicolor: petal width (cm)')
# Show the result
plt.show()


# Computing the covariance
# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)
# Print covariance matrix
print(covariance_matrix)
# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]
# Print the length/width covariance
print(petal_cov)


# Computing the Pearson correlation coefficient
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat  =np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)
# Print the result
print(r)










# CHapter 3: Probabilistic logic and statistical inference
# Generating random numbers using the np.random module
# Seed the random number generator
np.random.seed(42)
# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)
# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()
# Plot a histogram
_ = plt.hist(random_numbers)
# Show the plot
plt.show()


# The np.random module and Bernoulli trials
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0
    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()
        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success +=1
    return n_success



# How many defaults might we expect?
# Seed random number generator
np.random.seed(42)
# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)
# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)

# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed = True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')
# Show the plot
plt.show()



# Will the bank fail?
# Compute ECDF: x, y
x,y = ecdf(n_defaults) # function ecdf is defined above
# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaults')
_ = plt.ylabel('ECDF')
# Show the plot
plt.show()
# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)
# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))



# Sampling out of the Binomial distribution
# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n = 100, p = 0.05, size = 10000)
# Compute CDF: x, y
x, y = ecdf(n_defaults)
# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaults')
_ = plt.ylabel('ECDF')
# Show the plot
plt.show()



# Plotting the Binomial PMF
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults)  + 1.5) - 0.5
# Generate histogram
_ = plt.hist(n_defaults, normed=True, bins = bins)
# Label axes
_ = plt.xlabel('number of defaults')
_ = plt.ylabel('Probability mass function (PMF)')
# Show the plot
plt.show()



# Relationship between Binomial and Poisson distributions
# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size = 10000)
# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))
# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]
# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size = 10000)
    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))



# Was 2015 anomalous?
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size = 10000)
# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)
# Compute probability of getting seven or more: p_large
p_large = n_large / len(n_nohitters)
# Print the result
print('Probability of seven or more no-hitters:', p_large)













# Chapter 4: Thinking probabilistically-- Continuous variables
