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
from tensorflow.python.keras.optimizers import SGD


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







# Chapter 3: Neural Networks in TensorFlow
# The linear algebra of dense layers
borrower_features = pd.DataFrame(np.array([[20000.0, 2, 2,1,24,2,2,-1,-1,-2]]), 
                                 columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5'])
# Define inputs as a 32-bit float
inputs = np.array(borrower_features, np.float32)

# Initialize weights as 10x3 variable of ones
weights = tf.Variable(tf.ones((10, 3)))

# Perform matrix multiplication of the inputs by the weights
product = tf.matmul(inputs, weights)

# Apply sigmoid transformation
dense = tf.keras.activations.sigmoid(product)



# The low-level approach with multiple examples
borrower_features = tf.constant([[ 3.,  3., 23.],
       [ 2.,  1., 24.],
       [ 1.,  1., 49.],
       [ 1.,  1., 49.],
       [ 2.,  1., 29.]], tf.float32)

weights = tf.Variable([[-1.  ],
       [-2.  ],
       [ 0.05]])

# Compute the product of features and weights
products = tf.matmul(borrower_features, weights)

# Apply a sigmoid activation function
dense = keras.activations.sigmoid(products)

# Print products and dense tensors as numpy arrays
print(products.numpy())
print(dense.numpy())







# Using the dense layer operation
# Note that input data has been defined and is available as a 100x10 tensor: inputs. Additionally, keras.layers() is available.

data_path = 'uci_credit_card.csv'

# Load the dataset as a dataframe named housing 
uci_credit_card = pd.read_csv(data_path)
features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5']
inputs = uci_credit_card[features][:100]

# inputs muss noch in tensorflow konvertiert werden (tensorflow.python.framework.ops.EagerTensor).
# https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168
inputs = (
    tf.data.Dataset.from_tensor_slices(
        (tf.cast(uci_credit_card[features].values, tf.float32)))
    )

for features_tensor in inputs:
    print(f'features:{features_tensor}')
# inputs muss noch in tensorflow konvertiert werden (tensorflow.python.framework.ops.EagerTensor).


# Define the first dense layer
dense1 = tf.keras.layers.Dense(7, activation='sigmoid')(inputs)

# Define a dense layer with 3 output nodes
dense2 = tf.keras.layers.Dense(3, activation='sigmoid')(dense1)

# Define a dense layer with 1 output node
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

# Print dense layer 2 without using the numpy method
print(dense2)







# Binary classification problems
features = ['BILL_AMT1','BILL_AMT2','BILL_AMT3']
payments = uci_credit_card[features].values

# Construct input layer from features
input_layer = tf.constant(payments, tf.float32)

# Define first dense layer
dense_layer_1 = keras.layers.Dense(3, activation='relu')(input_layer)

# Define second dense layer
dense_layer_2 = keras.layers.Dense(2, activation='relu')(dense_layer_1)

# Define output layer
output_layer = keras.layers.Dense(1, activation='sigmoid')(dense_layer_2)
print(output_layer)




# Multiclass classification problems
features = ['BILL_AMT1','BILL_AMT2','BILL_AMT3', 'BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4']
borrower_features = uci_credit_card[features].values

# Construct input layer from borrower features
input_layer = tf.constant(borrower_features, tf.float32)

# Define first dense layer
dense_layer_1 = keras.layers.Dense(10, activation='sigmoid')(input_layer)

# Define second dense layer
dense_layer_2 = keras.layers.Dense(8, activation='relu')(dense_layer_1)

# Define output layer
output_layer = keras.layers.Dense(6, activation='softmax')(dense_layer_2)
print(output_layer)

# Knowing when to use sigmoid, relu, and softmax activations is an important step towards building and training neural networks in tensorflow.




# The dangers of local minima

initializer_1 = tf.Variable(5.0, tf.float32)
initializer_2 = tf.Variable(0.1, tf.float32)

# Define the optimization operation
opt = tf.keras.optimizers.SGD(lr=0.001)

for j in range(1000):
	# Perform minimization using the loss function and initializer_1
	opt.minimize(lambda: loss(initializer_1), var_list=[initializer_1])
	# Perform minimization using the loss function and initializer_2
	opt.minimize(lambda: loss(initializer_2), var_list=[initializer_2])

# Print initializer_1 and initializer_2 as numpy arrays
print(initializer_1.numpy(), initializer_2.numpy())



# NICHT LAUFFÄHIG!!!!
# Avoiding local minima
# Define the optimization operation for opt_1
opt_1 = keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.0)

# Define the optimization operation for opt_2
opt_2 = keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.99)

for j in range(100):
	opt_1.minimize(lambda: loss(momentum_1), var_list=[momentum_1])
    # Define the minimization operation for opt_2
	opt_2.minimize(lambda: loss(momentum_2), var_list=[momentum_2])

# Print momentum 1 and momentum 2 as numpy arrays
print(momentum_1.numpy(), momentum_2.numpy())



# Initialization in TensorFlow
# Define the layer 1 weights
weights1 = tf.Variable(tf.random.normal([23,7]))

# Initialize the layer 1 bias
bias1 = tf.Variable(tf.ones([7]))

# Define the layer 2 weights
weights2 = tf.Variable(tf.random.normal([7,1]))

# Define the layer 2 bias
bias2 = tf.Variable(0.0)



# NICHT LAUFFÄHIG!!!!

# Training neural networks with TensorFlow
# In this exercise, you will train a neural network to predict whether a credit card holder will default. The features and targets you will use to train your network are available in the Python shell as borrower_features and default. You defined the weights and biases in the previous exercise.
# Note that output_layer is defined as σ(layer1∗weights2+bias2), where σ
# is the sigmoid activation, layer1 is a tensor of nodes for the first hidden dense layer, weight2 is a tensor of weights, and bias2 is the bias tensor.
# The trainable variables are weights1, bias1, weights2, and bias2. Additionally, the following operations have been imported for you: nn.relu() and keras.layers.Dropout()
def loss_function(weights1, bias1, weights2, bias2, features, targets):
	# Apply relu activation functions to layer 1
	layer1 = nn.relu(add(matmul(features, weights1), bias1))
    # Apply dropout
	dropout = keras.layers.Dropout(0.25)(layer1)
	layer2 = nn.sigmoid(add(matmul(dropout, weights2), bias2))
    # Pass targets and layers2 to the cross entropy loss
	return keras.losses.binary_crossentropy(targets, layer2)
  
for j in range(0, 30000, 2000):
	features, targets = borrower_features[j:j+2000, :], default[j:j+2000, :]
    # Complete the optimizer
	opt.minimize(lambda: loss_function(weights1, bias1, weights2, bias2, features, targets), var_list=[weights1, bias1, weights2, bias2])
    
print(weights1.numpy())












# CHAPTER 4: High Level APIs in TensorFlow
# The sequential model in Keras
# Define a Keras sequential model
model = tf.keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the second dense layer
model.add(keras.layers.Dense(8, activation='relu'))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))




# Compiling a sequential model
# Define a Keras sequential model
model = keras.Sequential()

# Define the first dense layer
model.add(keras.layers.Dense(16, activation='sigmoid', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy')

# Print a model summary
print(model.summary())





# Defining a multiple input model
# For model 1, pass the input layer to layer 1 and layer 1 to layer 2
m1_layer1 = keras.layers.Dense(12, activation='sigmoid')(m1_inputs)
m1_layer2 = keras.layers.Dense(4, activation='softmax')(m1_layer1)

# For model 2, pass the input layer to layer 1 and layer 1 to layer 2
m2_layer1 = keras.layers.Dense(8, activation='relu')(m2_inputs)
m2_layer2 = keras.layers.Dense(4, activation='softmax')(m2_layer1)

# Merge model outputs
merged = keras.layers.add([m1_layer2, m2_layer2])

# Define functional model
model = keras.Model(inputs=[m1_inputs, m2_inputs], outputs=merged)



# NICHT LAUFFÄHIG, da die daten fehlen
# Training with Keras
# Define a sequential model
model = tf.keras.Sequential()

# Define a hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(784,)))

# Define the output layer
model.add(keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile('SGD', loss='categorical_crossentropy')

# Complete the fitting operation
model.fit(sign_language_features, sign_language_labels, epochs=5)




# Metrics and validation with Keras
# With the keras API, you only needed 14 lines of code to define, compile, train, and validate a model.
# Define sequential model
model = tf.keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(32, activation = 'sigmoid',  input_shape= (784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Set the optimizer, loss function, and metrics
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Add the number of epochs and the validation split
model.fit(sign_language_features, sign_language_labels, epochs=10, validation_split=0.1)






# Overfitting detection
# Define sequential model
model = keras.Sequential()

# Define the first layer
model.add(keras.layers.Dense(512, activation = 'relu',  input_shape= (784,)))

# Add activation function to classifier
model.add(keras.layers.Dense(4, activation='softmax'))

# Finish the model compilation
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

# Complete the model fit operation
model.fit(sign_language_features, sign_language_labels, epochs=50, validation_split=0.5)






# Evaluating models
# In practice, we often split the dataset into test and train sets.
# We then split the validation sample off from the train set using the validation_split parameter of model.fit().
# Evaluate the model using the train data
model.evaluate(train_features, train_labels)

# Evaluate the model using the test data
model.evaluate(test_features, test_labels)





# Preparing to train with Estimators
data_path = 'kc_house_data.csv'
housing = pd.read_csv(data_path)

# Define feature columns for bedrooms and bathrooms
bedrooms = tf.feature_column.numeric_column("bedrooms")
bathrooms = tf.feature_column.numeric_column("bathrooms")

# Define the list of feature columns
feature_list = [bedrooms, bathrooms]

def input_fn():
	# Define the labels
	labels = np.array(housing['price'])
	# Define the features
	features = {'bedrooms':np.array(housing['bedrooms']), 
                'bathrooms':np.array(housing['bathrooms'])}
	return features, labels



# Defining Estimators
# Define the model and set the number of steps
model = estimator.DNNRegressor(feature_columns=feature_list, hidden_units=[2,2])
model.train(input_fn, steps=1)

# Modify the code to use a LinearRegressor(), remove the hidden_units, and set the number of steps to 2.
# # Define the model and set the number of steps
model = estimator.LinearRegressor(feature_columns=feature_list)
model.train(input_fn, steps=2)
# Note that you have other premade estimator options, such as BoostedTreesRegressor(), and can also create your own custom estimators.

