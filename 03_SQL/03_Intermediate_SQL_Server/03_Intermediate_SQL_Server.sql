-- CHAPTER 1: Summarizing Data

use 03_intermediate_sql_server;
-- Calculate the average, minimum and maximum
SELECT AVG(DurationSeconds) AS Average, 
       MIN(DurationSeconds) AS Minimum, 
       MAX(DurationSeconds) AS Maximum
FROM incidents;

-- Calculate the aggregations by Shape
SELECT Shape,
       AVG(DurationSeconds) AS Average, 
       MIN(DurationSeconds) AS Minimum, 
       MAX(DurationSeconds) AS Maximum
FROM Incidents
GROUP BY Shape;

-- Calculate the aggregations by Shape
SELECT Shape,
       AVG(DurationSeconds) AS Average, 
       MIN(DurationSeconds) AS Minimum, 
       MAX(DurationSeconds) AS Maximum
FROM Incidents
GROUP BY Shape
-- Return records where minimum of DurationSeconds is greater than 1
HAVING MIN(DurationSeconds) > 1;


-- Removing missing values
-- Return the specified columns
SELECT IncidentDateTime, IncidentState
FROM Incidents
-- Exclude all the missing values from IncidentState  
WHERE IncidentState IS NOT NULL;


-- Imputing missing values (I)
-- Check the IncidentState column for missing values and replace them with the City column
-- SELECT IncidentState, ISNULL(IncidentState, City) AS Location -- in T-SQL

SELECT IncidentState, ifNull(IncidentState, City) AS Location
FROM Incidents
-- Filter to only return missing values from IncidentState
WHERE IncidentState IS NULL;

-- Imputing missing values (II)
-- Replace missing values 
SELECT Country, COALESCE(Country, IncidentState, City) AS Location
FROM Incidents
WHERE Country IS NULL;


-- Using CASE statements
SELECT Country, 
       CASE WHEN Country = 'us'  THEN 'USA'
       ELSE 'International'
       END AS SourceCountry
FROM Incidents;

-- Creating several groups with CASE
-- Complete the syntax for cutting the duration into different cases
SELECT DurationSeconds, 
-- Start with the 2 TSQL keywords, and after the condition a TSQL word and a value
      CASE WHEN (DurationSeconds <= 120) THEN 1
-- The pattern repeats with the same keyword and after the condition the same word and next value          
       WHEN (DurationSeconds > 120 AND DurationSeconds <= 600) THEN 2
-- Use the same syntax here             
       WHEN (DurationSeconds > 601 AND DurationSeconds <= 1200) THEN 3
-- Use the same syntax here               
       WHEN (DurationSeconds > 1201 AND DurationSeconds <= 5000) THEN 4
-- Specify a value      
       ELSE 5 
       END AS SecondGroup   
FROM Incidents;


-- CHAPTER 2: Math Functions
-- Calculating the total
-- Write a query that returns an aggregation 
SELECT MixDesc, SUM(Quantity) AS Total
FROM mixdata
-- Group by the relevant column
GROUP By MixDesc;

-- Counting the number of rows
-- Count the number of rows by MixDesc
SELECT MixDesc, COUNT(*)
FROM mixdata
GROUP BY MixDesc;

-- Counting the number of days between dates
-- Return the difference in OrderDate and ShipDate
SELECT OrderDate, ShipDate, 
       -- DATEDIFF(DD, OrderDate, ShipDate) AS Duration / T-SQL
       DATEDIFF(OrderDate, ShipDate) AS Duration
FROM mixdata;

-- Adding days to a date
-- Return the DeliveryDate as 5 days after the ShipDate
SELECT OrderDate, 
       -- DATEADD(DD, 5, ShipDate) AS DeliveryDate
       DATE_ADD(ShipDate, INTERVAL 5 DAY) AS DeliveryDate
FROM mixdata;

-- Rounding numbers
-- Round Cost to the nearest dollar
SELECT Cost, 
       ROUND(Cost, 0) AS RoundedCost
FROM mixdata;


-- Truncating numbers
-- Truncate cost to whole number
SELECT Cost, 
       -- ROUND(Cost, 0, 1) AS TruncateCost / T-SQL
		  TRUnCATE(Cost, 0) AS TruncateCost
FROM mixdata;


-- Calculating the absolute value
-- Return the absolute value of DeliveryWeight
SELECT DeliveryWeight,
       ABS(DeliveryWeight) AS AbsoluteValue
FROM mixdata;


-- Calculating squares and square roots
-- Return the square and square root of WeightValue
SELECT WeightValue, 
       -- SQUARE(WeightValue) AS WeightSquare, / T-SQL
       POWER(WeightValue, 2) AS WeightSquare, 
       SQRT(WeightValue) AS WeightSqrt
FROM mixdata;