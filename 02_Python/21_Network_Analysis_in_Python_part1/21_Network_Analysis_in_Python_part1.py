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
import datetime

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




# Visualizing using Matrix plots
# Import nxviz
import nxviz as nv
# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)
# Draw m to the screen
m.draw()
# Display the plot
plt.show()
# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)
# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()
    
    
    

# Visualizing using Circos plots
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot
# Create the CircosPlot object: c
c = CircosPlot(T)
# Draw c to the screen
c.draw()
# Display the plot
plt.show()



# Visualizing using Arc plots
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot
# Create the un-customized ArcPlot object: a
a = ArcPlot(T)
# Draw a to the screen
a.draw()
# Display the plot
plt.show()
# Create the customized ArcPlot object: a2
a2 = ArcPlot(T, node_order='category', node_color='category')
# Draw a2 to the screen
a2.draw()
# Display the plot
plt.show()






# Chapter 2: Important nodes
# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    # Iterate over all nodes in G
    for n in G.nodes():
        # Check if the number of neighbors of n matches m
        if len(G.neighbors(n)) == m:
            # Add the node n to the set
            nodes.add(n)
    # Return the nodes with m neighbors
    return nodes
# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)




# Compute degree distribution
# Compute the degree of every node: degrees
degrees = [len(T.neighbors(n)) for n in T.nodes()]
# Print the degrees
print(degrees)




# Degree centrality distribution
# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)
# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()
# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist([len(T.neighbors(n)) for n in T.nodes()])
plt.show()
# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.show()




# Shortest Path I
# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    # Initialize the queue of cells to visit with the first node: queue
    queue = [node1]
    # Iterate over the nodes in the queue
    for node in queue:
        # Get neighbors of the node
        neighbors = G.neighbors(node)
        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        



# Shortest Path II
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    
    for node in queue:  
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        
        else:
            # Add current node to visited nodes
            visited_nodes.add(node)
            
            # Add neighbors of current node that have not yet been visited
            queue.extend([n for n in neighbors if n not in visited_nodes])
            


            
# Shortest Path III
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    queue = [node1]
    
    for node in queue:  
        neighbors = G.neighbors(node)
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break

        else:
            visited_nodes.add(node)
            queue.extend([n for n in neighbors if n not in visited_nodes])
        
        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))

            # Place the appropriate return statement
            return False
        
        
        
# NetworkX betweenness centrality on a social network
# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)
# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)
# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()))
# Display the plot
plt.show()



# Deep dive - Twitter network
# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):
    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))
    nodes = set()
    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
        # Check if the current value has the maximum degree centrality
        if v == max_dc:
            # Add the current node to the set of nodes
            nodes.add(k)
    return nodes
    
# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)

# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())
    
    
    
    
    
# Deep dive - Twitter network part II
 # Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):
    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)
    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))
    nodes = set()
    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():
        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:
            # Add the current node to the set of nodes
            nodes.add(k)
    return nodes

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)

# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())   