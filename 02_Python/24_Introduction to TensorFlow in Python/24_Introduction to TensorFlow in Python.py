# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:02:39 2019

@author: d91067
"""

# import inspect
# lines = inspect.getsource(foo)
# print(lines)

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import constant, add
from tensorflow import keras
from tensorflow.python.keras.optimizers import Adam
# from tf.keras.optimizers.Adam import Adam, SGD


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
import matplotlib.pyplot as plt



plt.style.use('ggplot')
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\24_Introduction to TensorFlow in Python'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\24_Introduction to TensorFlow in Python'
os.chdir(path)



# Chapter 1: Introduction to TensorFlow
# Defining constants with convenience functions
# Define a 3x4 tensor with all values equal to 9
x = tf.fill([3, 4], 9)

# Define a tensor of ones with the same shape as x
y = tf.ones_like(x)

# Define the one-dimensional vector, z
z = tf.constant([1, 2, 3, 4])

# Print z as a numpy array
print(z.numpy())



# Defining variables
# Define the 1-dimensional variable X
X = tf.Variable([1, 2, 3, 4])

# Print the variable X
print(X)

# Convert X to a numpy array and assign it to Z
Z = X.numpy()

# Print Z
print(Z)



# Checking properties of tensors
print(X)
print(Z)




# Performing element-wise multiplication
# Define tensors A0 and B0 as constants
A0 = constant([1, 2, 3, 4])
B0 = constant([[1, 2, 3], [1, 6, 4]])

# Define A1 and B1 to have the correct shape
A1 = tf.ones_like(4,)
B1 = tf.ones_like(2,3)

# Perform element-wise multiplication
A2 = tf.multiply(A0,A1)
B2 = tf.multiply(B0,B1)

# Print the tensors A2 and B2
print(A2.numpy())
print(B2.numpy())





# Making predictions with matrix multiplication
# Define X, b, and y as constants
X = constant([[1, 2], [2, 1], [5, 8], [6, 10]])
b = constant([[1], [2]])
y = constant([[6], [4], [20], [23]])

# Compute ypred using X and b
ypred = tf.matmul(X,b)

# Compute and print the error as y - ypred
error = tf.subtract(y, ypred)
print(error.numpy())



# Summing over tensor dimensions
wealth = constant([[11, 7, 4, 3, 25], [50, 2, 60, 0, 10]])
tf.reduce_sum(wealth)
tf.reduce_sum(wealth, 0)
tf.reduce_sum(wealth, 1)





# Reshaping tensors
# Define input image
image = tf.ones([16, 16])

# Reshape image into a vector
image_vector = tf.reshape(image, (256, 1))

# Reshape image into a higher dimensional tensor
image_tensor = tf.reshape(image, (4, 4, 4, 4))


# Add three color channels
image = tf.ones([16, 16, 3])

# Reshape image into a vector
image_vector = tf.reshape(image, (768, 1))

# Reshape image into a higher dimensional tensor
image_tensor = tf.reshape(image, (4, 4, 4, 4, 3))





# Optimizing with gradients
# Define x as a variable equal to 0.0
x = tf.Variable([0.0])

# Define y using the multiply operation and apply Gradient Tape
with tf.GradientTape() as tape:
	tape.watch(x)
	y = tf.multiply(x,x)
	
# Compute the gradient of y with respect to x
g = tape.gradient(y, x)

# Compute and print the gradient using the numpy method
print(g.numpy())




# Performing graph-based computations
model = constant([[ 1.,  0., -1.]])
letter = constant([[1., 0., 1.],
       [1., 1., 0.],
       [1., 0., 1.]])
# Reshape model from a 1x3 to a 3x1 tensor
model = tf.reshape(model, (3, 1))

# Multiply letter by model
output = tf.matmul(letter, model)

# Sum over output and print prediction using the numpy method
prediction = tf.reduce_sum(output)
print(prediction.numpy())


















# CHAPTER 2: Linear Regression in TensorFlow
# Load data using pandas
# Assign the path to a string variable named data_path
data_path = 'kc_house_data.csv'

# Load the dataset as a dataframe named housing 
housing = pd.read_csv(data_path)
housing['price_log'] = np.log(housing['price'])
housing['lot_size_log'] = np.log(housing['sqft_lot'])

# Print the price column of housing
print(housing['price'])



# Bringing everything together
# Use a numpy array to define price as a 32-bit float
price = np.array(housing['price'], np.float32)

# Define waterfront as a Boolean using cast
waterfront = tf.cast(housing['waterfront'], tf.bool)

# Print price and waterfront
print(price)
print(waterfront)



# Loss functions in TensorFlow
# Import the keras module from tensorflow
from tensorflow import keras 

# Compute the mean squared error (mse)
loss = keras.losses.mse(price, predictions)
loss = keras.losses.mae(price, predictions)

# Print the mean squared error (mse)
print(loss.numpy())






# Modifying the loss function
# Initialize a variable named scalar
scalar = Variable(1.0, float32)

# Define a loss function
def loss_function(scalar, features, target):
	# Define the predicted values
	predictions = scalar*features
	# Return the MAE loss
	return keras.losses.mae(target, predictions)

# Evaluate and print the loss function
print(loss_function(scalar, features, target).numpy())





# Setup a linear regression
# Define the intercept and slope
intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope):
	# Set the predicted values
	pred_price_log = intercept + slope * lot_size_log
    # Return the MSE loss
	return keras.losses.mse(price_log, pred_price_log)

# Initialize an adam optimizer
opt = tf.keras.optimizers.Adam(0.1)




# Train a linear model
for j in range(500):
	# Apply minimize, pass the loss function, and supply the variables
    opt.minimize(lambda: loss_function(intercept, slope), var_list=[intercept, slope])
    if j % 100 == 0:
  	  print(loss_function(intercept, slope).numpy())

# Print the intercept and slope as numpy arrays
print(intercept.numpy(), slope.numpy())




# Multiple linear regression
# Define variables for intercept, slope_1, and slope_2
intercept = tf.Variable(0.1, tf.float32)
slope_1 = tf.Variable(0.1, tf.float32)
slope_2 = tf.Variable(0.1, tf.float32)

# Define the loss function
def loss_function(intercept, slope_1, slope_2):
	# Use the mean absolute error loss
	return keras.losses.mae(price_log, intercept+lot_size_log*slope_1+bedrooms*slope_2)

# Define the optimize operation
opt = keras.optimizers.Adam()

# Perform one minimization step
opt.minimize(lambda: loss_function(intercept, slope_1, slope_2), var_list=[intercept, slope_1, slope_2])



# Preparing to batch train
# Define the intercept and slope
intercept =tf. Variable(10.0, tf.float32)
slope = tf.Variable(0.5, tf.float32)

# Define the loss function
def loss_function (intercept, slope, features, target):
	# Define the predicted values
	predictions = intercept + slope * features
 	# Define the MSE loss
	return keras.losses.mse(target, predictions)




# Training a linear model in batches
# Initialize adam optimizer
opt = keras.optimizers.Adam()

# Load data in batches
for batch in pd.read_csv('kc_house_data.csv', chunksize=100):
	size_batch = np.array(batch['sqft_lot'], np.float32)
	# Extract the price values for the current batch
	price_batch = np.array(batch['price'], np.float32)
	# Complete the loss, fill in the variable list, and minimize
	opt.minimize(lambda: loss_function(intercept, slope, size_batch, price_batch), var_list=[intercept, slope])

# Print trained parameters
print(intercept.numpy(), slope.numpy())





