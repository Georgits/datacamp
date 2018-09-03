# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:22:49 2017

@author: d91067
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = 'C:\\Users\\d91067\\Desktop\\R\\datacamp\\02_Python\\10_Introduction_to_Databases_in_Python'
# path = 'C:\\Users\\georg\\Desktop\\georgi\\github\\datacamp\\02_Python\\10_Introduction_to_Databases_in_Python'
os.chdir(path)


# Chapter 1: Basics of Relational Databases
# Engines and Connection Strings
# Import create_engine
from sqlalchemy import create_engine
# Create an engine that connects to the census.sqlite file: engine
engine = create_engine('sqlite:///census.sqlite')
# Print table names
print(engine.table_names())


# Autoloading Tables from a Database
# Import Table
from sqlalchemy import MetaData, Table
metadata = MetaData()
# Reflect census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Print census table metadata
print(repr(census))


# Viewing Table Details
# Reflect the census table from the engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Print the column names
print(census.columns.keys())
# Print full table metadata
print(repr(metadata.tables['census']))


# Selecting data from a Table: raw SQL
engine = create_engine('sqlite:///census.sqlite')
from sqlalchemy import create_engine
connection = engine.connect()
# Build select statement for census table: stmt
stmt = 'SELECT * from census'
# Execute the statement and fetch the results: results
results = connection.execute(stmt).fetchall()
# Print results
print(results)


# Selecting data from a Table with SQLAlchemy
# Import select
from sqlalchemy import select
# Reflect census table via engine: census
census = Table('census', metadata, autoload=True, autoload_with=engine)
# Build select statement for census table: stmt
stmt = select([census])
# Print the emitted statement to see the SQL emitted
print(stmt)
# Execute the statement and print the results
print(connection.execute(stmt).fetchall())


# Handling a ResultSet
# Get the first row of the results by using an index: first_row
first_row = results[0]
# Print the first row of the results
print(first_row)
# Print the first column of the first row by using an index
print(first_row[0])
# Print the 'state' column of the first row by using its name
print(first_row['state'])









# Chapter 2: Applying Filtering, Ordering and Grouping to Queries
# Connecting to a PostgreSQL Database
# Import create_engine function
from sqlalchemy import create_engine
# Create an engine to the census database
# There are three components to the connection string in this exercise: the dialect and driver ('postgresql+psycopg2://'), 
# followed by the username and password ('student:datacamp'), followed by the host and port ('@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/'), and finally, 
# the database name ('census'). You will have to pass this string as an argument to create_engine() in order to connect to the database.
engine = create_engine('postgresql+psycopg2://student:datacamp@postgresql.csrrinzqubik.us-east-1.rds.amazonaws.com:5432/census')
# Use the .table_names() method on the engine to print the table names
print(engine.table_names())


# Filter data selected from a Table - Simple
# Create a select query: stmt
stmt = select([census])
# Add a where clause to filter the results to only those for New York
stmt = stmt.where(census.columns.state == 'New York')
# Execute the query to retrieve all the data returned: results
results = connection.execute(stmt).fetchall()
# Loop over the results and print the age, sex, and pop2008
for result in results:
    print(result.age, result.sex, result.pop2008)



# Filter data selected from a Table - Expressions
states =  ['New York', 'California', 'Texas']
# Create a query for the census table: stmt
stmt = select([census])
# Append a where clause to match all the states in_ the list states
stmt = stmt.where(census.columns.state.in_(states))
# Loop over the ResultProxy and print the state and its population in 2000
for result in connection.execute(stmt):
    print(result.state, result.pop2000)


# Filter data selected from a Table - Advanced
# Import and_
from sqlalchemy import and_
# Build a query for the census table: stmt
stmt = select([census])
# Append a where clause to select only non-male records from California using and_
stmt = stmt.where(
    # The state of California with a non-male sex
    and_(census.columns.state == 'California',
         census.columns.sex != 'M'
         )
)
# Loop over the ResultProxy printing the age and sex
for result in connection.execute(stmt):
    print(result.age, result.sex)




# Ordering by a Single Column
# Build a query to select the state column: stmt
stmt = select([census.columns.state])
# Order stmt by the state column
stmt = stmt.order_by(census.columns.state)
# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()
# Print the first 10 results
print(results[:10])



# Ordering in Descending Order by a Single Column
# Import desc
from sqlalchemy import desc
# Build a query to select the state column: stmt
stmt = select([census.columns.state])
# Order stmt by state in descending order: rev_stmt
rev_stmt = stmt.order_by(desc(census.columns.state))
# Execute the query and store the results: rev_results
rev_results = connection.execute(rev_stmt).fetchall()
# Print the first 10 rev_results
print(rev_results[:10])



# Ordering by Multiple Columns
# Build a query to select state and age: stmt
stmt = select([census.columns.state, census.columns.age])
# Append order by to ascend by state and descend by age
stmt = stmt.order_by(census.columns.state, desc(census.columns.age))
# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()
# Print the first 20 results
print(results[:20])



# Counting Distinct Data
from sqlalchemy import func
# Build a query to count the distinct states values: stmt
stmt = select([func.count(census.columns.state.distinct())])
# Execute the query and store the scalar result: distinct_state_count
distinct_state_count = connection.execute(stmt).scalar()
# Print the distinct_state_count
print(distinct_state_count)



# Count of Records by State
# Import func
from sqlalchemy import func
# Build a query to select the state and count of ages by state: stmt
stmt = select([census.columns.state, func.count(census.columns.age)])
# Group stmt by state
stmt = stmt.group_by(census.columns.state)
# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()
# Print results
print(results)
# Print the keys/column names of the results returned
print(results[0].keys())



# Determining the Population Sum by State
# Import func
from sqlalchemy import func
# Build an expression to calculate the sum of pop2008 labeled as population
pop2008_sum = func.sum(census.columns.pop2008).label('population')
# Build a query to select the state and sum of pop2008: stmt
stmt = select([census.columns.state, pop2008_sum])
# Group stmt by state
stmt = stmt.group_by(census.columns.state)
# Execute the statement and store all the records: results
results = connection.execute(stmt).fetchall()
# Print results
print(results)
# Print the keys/column names of the results returned
print(results[0].keys())



# SQLAlchemy ResultsProxy and Pandas Dataframes
# import pandas
import pandas as pd
# Create a DataFrame from the results: df
df = pd.DataFrame(results)
# Set column names
df.columns = results[0].keys()
# Print the Dataframe
print(df)


# From SQLAlchemy results to a Graph
# Import Pyplot as plt from matplotlib
import matplotlib.pyplot as plt
# Create a DataFrame from the results: df
df = pd.DataFrame(results)
# Set Column names
df.columns = results[0].keys()
# Print the DataFrame
print(df)
# Plot the DataFrame
df.plot.bar()
plt.show()









# Chapter 4: Advanced SQLAlchemy Queries
