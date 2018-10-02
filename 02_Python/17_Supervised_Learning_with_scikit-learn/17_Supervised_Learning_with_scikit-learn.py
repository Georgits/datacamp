# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 20:57:58 2018

@author: georg
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
from sklearn import datasets
plt.style.use('ggplot')
# path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\17_Supervised_Learning_with_scikit-learn'
path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\17_Supervised_Learning_with_scikit-learn'
os.chdir(path)




# Iris-Dataset
iris = datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data), type(iris.target)
iris.data.shape
iris.target_names

X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head())
_ = pd.scatter_matrix(df, c = y, figsize = [8, 8], s=150, marker = 'D')




# Chapter 1: Classification
df.head()
df.info()
df.describe()
# 'education' und 'party' sind 0/1-Variablen
# NICHt LAUFFÄHIG, da die daten fehlen
plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()



# k-Nearest Neighbors: Fit
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values
# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X, y)



# k-Nearest Neighbors: Predict
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values
# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the classifier to the data
knn.fit(X, y)
# Predict the labels for the training data X
y_pred = knn.predict(X)
# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

