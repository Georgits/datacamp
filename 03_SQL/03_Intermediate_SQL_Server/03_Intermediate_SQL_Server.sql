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



-- CHAPTER 3: Processing Data in SQL Server
-- In diesem Chapter ist der Code in T-SQL
-- Creating and using variables
-- Declare the variable (a SQL Command, the var name, the datatype)

-- T - SQL
DECLARE @counter INT 
-- Set the counter to 20
SET @counter = 20
-- Select and increment the counter by one 
-- Print the variable
SELECT @counter

-- T - SQL
-- Creating and using variables
-- Declare the variable (a SQL Command, the var name, the datatype)
DECLARE @counter INT 
-- Set the counter to 20
SET @counter = 20
-- Select and increment the counter by one 
SET @counter = @counter + 1
-- Print the variable
SELECT @counter


-- Creating a WHILE loop
DECLARE @counter INT 
SET @counter = 20

-- Create a loop
WHILE @counter < 30

-- Loop code starting point
BEGIN
	SELECT @counter = @counter + 1
-- Loop finish
END

-- Check the value of the variable
SELECT @counter

-- Queries with derived tables (I)
SELECT a.RecordId, a.Age, a.BloodGlucoseRandom, 
-- Select maximum glucose value (use colname from derived table)    
       b.MaxGlucose
FROM Kidney a
-- Join to derived table
JOIN (SELECT Age, MAX(BloodGlucoseRandom) AS MaxGlucose FROM Kidney GROUP BY Age) b
-- Join on Age
ON a.Age = b.Age

-- Queries with derived tables (II)
-- In this exercise, you will create a derived table to return all patient records with the highest BloodPressure at their Age level.
SELECT *
FROM Kidney a
-- Create derived table: select age, max blood pressure from kidney grouped by age
JOIN (SELECT Age, MAX(BloodPressure) AS MaxBloodPressure FROM kidney GROUP BY Age) b
-- JOIN on BloodPressure equal to MaxBloodPressure
ON a.BloodPressure = b.MaxBloodPressure
-- Join on Age
AND a.Age = b.Age


-- Creating CTEs (I)
-- Specify the keyowrds to create the CTE
WITH BloodGlucoseRandom (MaxGlucose) 
AS (SELECT MAX(BloodGlucoseRandom) AS MaxGlucose FROM Kidney)

SELECT a.Age, b.MaxGlucose
FROM Kidney a
-- Join the CTE on blood glucose equal to max blood glucose
JOIN BloodGlucoseRandom b
ON a.BloodGlucoseRandom = b.MaxGlucose;


-- Creating CTEs (II)
-- Create the CTE
WITH BloodPressure 
AS (SELECT MAX(BloodPressure) AS MaxBloodPressure FROM Kidney)

SELECT *
FROM Kidney a
-- Join the CTE  
JOIN BloodPressure b
ON a.BloodPressure = b.MaxBloodPressure;




-- CHAPTER 4: Window Functions
-- Window functions with aggregations (I)
SELECT OrderID, TerritoryName, 
       -- Total price for each partition
       SUM(OrderPrice) 
       -- Create the window and partitions
       OVER (PARTITION BY TerritoryName) AS TotalPrice
FROM orders;


-- Window functions with aggregations (II)
SELECT OrderID, TerritoryName, 
       -- Number of rows per partition
       COUNT(*) 
       -- Create the window and partitions
       OVER (PARTITION BY TerritoryName) AS TotalOrders
FROM orders;


-- First value in a window
SELECT TerritoryName, OrderDate, 
       -- Select the first value in each partition
       FIRST_VALUE(OrderDate) 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS FirstOrder
FROM orders;


-- Previous and next values
SELECT TerritoryName, OrderDate, 
       -- Specify the previous OrderDate in the window
       LAG(OrderDate) 
       -- Over the window, partition by territory & order by order date
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS PreviousOrder,
       -- Specify the next OrderDate in the window
       LEAD(OrderDate) 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS NextOrder
FROM orders;


-- Creating running totals
-- You usually don't have to use ORDER BY when using aggregations, but if you want to create running totals, you should arrange your rows!
-- In this exercise, you will create a running total of OrderPrice
SELECT TerritoryName, OrderDate, 
       -- Create a running total
       SUM(OrderPrice) 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS TerritoryTotal	  
FROM orders;


-- Assigning row numbers
SELECT TerritoryName, OrderDate, 
       -- Assign a row number
       ROW_NUMBER() 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS OrderCount
FROM orders;

SELECT TerritoryName, OrderDate, 
       -- Assign a row number
       ROW_NUMBER() 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS OrderCount,
              -- Create a running total
       SUM(OrderPrice) 
       -- Create the partitions and arrange the rows
       OVER(PARTITION BY TerritoryName ORDER BY OrderDate) AS TerritoryTotal	  
FROM orders;


-- Calculating standard deviation
SELECT OrderDate, TerritoryName, 
       -- Calculate the standard deviation
	   -- STDEV(OrderPrice) / T-SQL
   	   STDDEV(OrderPrice) 
       OVER (PARTITION BY TerritoryName ORDER BY OrderDate) AS StdDevPrice	  
FROM orders;


-- Calculating mode (I)
-- T-SQL
-- Create a CTE Called ModePrice which contains two columns
WITH ModePrice (OrderPrice, UnitPriceFrequency)
AS
(
	SELECT OrderPrice, 
	ROW_NUMBER() 
	OVER(PARTITION BY OrderPrice ORDER BY OrderPrice) AS UnitPriceFrequency
	FROM Orders 
)

-- Select everything from the CTE
SELECT * FROM ModePrice ;



-- T-SQL
-- Calculating mode (II)
-- CTE from the previous exercise
WITH ModePrice (OrderPrice, UnitPriceFrequency)
AS
(
	SELECT OrderPrice,
	ROW_NUMBER() 
    OVER (PARTITION BY OrderPrice ORDER BY OrderPrice) AS UnitPriceFrequency
	FROM Orders
)

-- Select the order price from the CTE
SELECT OrderPrice AS ModeOrderPrice
FROM ModePrice
-- Select the maximum UnitPriceFrequency from the CTE
WHERE UnitPriceFrequency IN (SELECT MAX(UnitPriceFrequency) FROM ModePrice);