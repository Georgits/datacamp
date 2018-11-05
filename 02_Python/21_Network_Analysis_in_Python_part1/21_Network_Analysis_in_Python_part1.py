# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:52:34 2018

@author: georg
"""



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import random
import matplotlib.pyplot as plt
import networkx as nx


plt.style.use('ggplot')
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\21_Network_Analysis_in_Python_part1'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\21_Network_Analysis_in_Python_part1'
os.chdir(path)






# Chapter 1: Introduction to networks
# Basics of NetworkX API, using Twitter network

import networkx as nx
import requests
import date

# Read the graph from local disk.
T = nx.read_gpickle('ego-twitter.p')
__T = nx.read_gpickle('ego-twitter.p')

# Basics of NetworkX API, using Twitter network
type(T.nodes())

len(T.nodes())

T.edges(data=True)[-1]



# Basic drawing of a network using NetworkX
# NICHT LAUFFÄHIG
# Draw the graph to screen
nx.draw(T_sub)
plt.show()


# Queries on a graph
# Use a list comprehension to get the nodes of interest: noi
T.nodes(data=True)[-1]
noi = [n for n, d in T.nodes(data = True) if d['occupation'] == 'scientist']

# Use a list comprehension to get the edges of interest: eoi
T.edges(data=True)[-1]
eoi = [(u, v) for u, v, d in T.edges(data=True)  if d['date'] < datetime.date(2010, 1, 1)]



# Checking the un/directed status of a graph
type(T) # Directed Graph.


# Specifying a weight on edges
T.edge[1][10]
T.edge[10][1]


# Specifying a weight on edges
#  Refer to the following template to set an attribute of an edge: network_name.edge[node1][node2]['attribute'] = value
# Set the weight of the edge
T.edge[1][10]['weight'] = 2

# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data = True):

    # Check if node 293 is involved
    if 293 in [u,v]:
    
        # Set the weight to 1.1
        T[u][v]['weight'] = 1.1
        
        
# Checking whether there are self-loops in the graph
# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    # Iterate over all the edges of G
    for u, v in G.edges():
    # Check if node u and node v are the same
        if u == v:
            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)
    return nodes_in_selfloops
# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))