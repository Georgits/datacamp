use 05_writing_functions_and_stored_procedures_in_sql_server;

-- Das ist ein T-SQL Script und funktionniert ohne Anpassung in MySQL nicht
-- Wie das geht, s. beispielhaft direkt in nachfolgenden Zeilen

-- https://s3.amazonaws.com/capitalbikeshare-data/index.html

-- Chapter 1:  Temporal EDA, Variables & Date Manipulation


-- T-SQL
-- Transactions per day
SELECT
  -- Select the date portion of StartDate
  CONVERT(DATE, StartDate) as StartDate,
  -- Measure how many records exist for each StartDate
  COUNT(ID) as CountOfRows 
FROM CapitalBikeShare 
-- Group by the date portion of StartDate
GROUP BY CONVERT(DATE, StartDate)
-- Sort the results by the date portion of StartDate
ORDER BY CONVERT(DATE, StartDate);



-- MySQL
SELECT
  -- Select the date portion of StartDate
  CONVERT(StartDate, DATE) as StartDate,
  -- Measure how many records exist for each StartDate
  COUNT(*) as CountOfRows 
FROM CapitalBikeShare 
-- Group by the date portion of StartDate
GROUP BY CONVERT(StartDate, DATE)
-- Sort the results by the date portion of StartDate
ORDER BY CONVERT(StartDate, DATE);



-- Ab hier wieder T-SQL
-- Seconds or no seconds?
SELECT
	-- Count the number of IDs
	COUNT(ID) AS Count,
    -- Use DATEPART() to evaluate the SECOND part of StartDate
    "StartDate" = CASE WHEN DATEPART(SECOND, StartDate) = 0 THEN 'SECONDS = 0'
					  WHEN DATEPART(SECOND, StartDate) > 0 THEN 'SECONDS > 0' END
FROM CapitalBikeShare
GROUP BY
    -- Complete the CASE statement
	CASE WHEN DATEPART(SECOND, StartDate) = 0 THEN 'SECONDS = 0'
		WHEN DATEPART(SECOND, StartDate) > 0 THEN 'SECONDS > 0' END
		

		
		
-- Which day of week is busiest?
SELECT
    -- Select the day of week value for StartDate
	DATENAME(WEEKDAY, StartDate) as DayOfWeek,
    -- Calculate TotalTripHours
	SUM(DATEDIFF(SECOND, StartDate, EndDate))/ 3600 as TotalTripHours 
FROM CapitalBikeShare 
-- Group by the day of week
GROUP BY DATENAME(WEEKDAY, StartDate)
-- Order TotalTripHours in descending order
ORDER BY TotalTripHours DESC




-- Find the outliers
SELECT
	-- Calculate TotalRideHours using SUM() and DATEDIFF()
  	SUM(DATEDIFF(SECOND, StartDate, EndDate))/ 3600 AS TotalRideHours,
    -- Select the DATE portion of StartDate
  	CONVERT(DATE, StartDate) AS DateOnly,
    -- Select the WEEKDAY
  	DATENAME(WEEKDAY, CONVERT(DATE, StartDate)) AS DayOfWeek 
FROM CapitalBikeShare
-- Only include Saturday
WHERE DATENAME(WEEKDAY, StartDate) = 'Saturday' 
GROUP BY CONVERT(DATE, StartDate);





-- DECLARE & CAST
-- Create @ShiftStartTime
DECLARE @ShiftStartTime AS time = '08:00 AM'

-- Create @StartDate
DECLARE @StartDate AS date

-- Set StartDate to the first StartDate from CapitalBikeShare
SET 
	@StartDate = (
    	SELECT TOP 1 StartDate 
    	FROM CapitalBikeShare 
    	ORDER BY StartDate ASC
		)

-- Create ShiftStartDateTime
DECLARE @ShiftStartDateTime AS datetime

-- Cast StartDate and ShiftStartTime to datetime data types
SET @ShiftStartDateTime = CAST(@StartDate AS datetime) + CAST(@ShiftStartTime AS datetime) 

SELECT @ShiftStartDateTime




-- DECLARE a TABLE
-- Create @Shifts
DECLARE @Shifts TABLE(
    -- Create StartDateTime column
	StartDateTime datetime,
    -- Create EndDateTime column
	EndDateTime datetime)
-- Populate @Shifts
INSERT INTO @Shifts (StartDateTime, EndDateTime)
	SELECT '3/1/2018 8:00 AM', '3/1/2018 4:00 PM'
SELECT * 
FROM @Shifts



-- INSERT INTO @TABLE
-- Create @RideDates
DECLARE @RideDates TABLE(
    -- Create RideStart
	RideStart date,
    -- Create RideEnd
	RideEnd date)
-- Populate @RideDates
INSERT INTO @RideDates(RideStart, RideEnd)
-- Select the unique date values of StartDate and EndDate
SELECT DISTINCT
    -- Cast StartDate as date
	CAST(StartDate AS date),
    -- Cast EndDate as date
	CAST(EndDate AS date) 
FROM CapitalBikeShare 
SELECT * 
FROM @RideDates;




-- Parameters matter with DATEDIFF
SELECT DATEDIFF(day, '2/26/2018', '3/3/2018')
SELECT DATEDIFF(week, '2/26/2018', '3/3/2018')
SELECT DATEDIFF(month, '2/26/2018', '3/3/2018')


-- First day of month
-- Find the first day of the current month
SELECT DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)



-- Chapter 2: User Defined Functions
-- What was yesterday?
-- Create GetYesterday()
CREATE FUNCTION GetYesterday()
-- Specify return data type
RETURNS date AS BEGIN
-- Calculate yesterday's date value
RETURN (SELECT DATEADD(day, -1, GETDATE()))
END;

-- One in one out
-- Create SumRideHrsSingleDay
CREATE FUNCTION SumRideHrsSingleDay (@DateParm date)
-- Specify return data type
RETURNS numeric
AS
-- Begin
BEGIN
RETURN
-- Add the difference between StartDate and EndDate
(SELECT SUM(DATEDIFF(second, StartDate, EndDate))/3600
FROM CapitalBikeShare
 -- Only include transactions where StartDate = @DateParm
WHERE CAST(StartDate AS date) = @DateParm)
-- End
END;



-- Multiple inputs one output
-- Create the function
CREATE FUNCTION SumRideHrsDateRange (@StartDateParm datetime, @EndDateParm datetime)
-- Specify return data type
RETURNS numeric
AS
BEGIN
RETURN
-- Sum the difference between StartDate and EndDate
(SELECT SUM(DATEDIFF(second, StartDate, EndDate))/3600
FROM CapitalBikeShare
-- Include only the relevant transactions
WHERE StartDate > @StartDateParm and StartDate < @EndDateParm)
END;


-- Inline TVF
-- Create the function
CREATE FUNCTION SumStationStats(@StartDate AS datetime)
-- Specify return data type
RETURNS TABLE
AS
RETURN
SELECT
	StartStation,
    -- Use COUNT() to select RideCount
	COUNT(ID) AS RideCount,
    -- Use SUM() to calculate TotalDuration
    SUM(DURATION) AS TotalDuration
FROM CapitalBikeShare
WHERE CAST(StartDate as Date) = @StartDate
-- Group by StartStation
GROUP BY StartStation;



-- Multi statement TVF
-- Create the function
CREATE FUNCTION CountTripAvgDuration (@Month CHAR(2), @Year CHAR(4))
-- Specify return variable
RETURNS @DailyTripStats TABLE(
	TripDate	date,
	TripCount	int,
	AvgDuration	numeric)
AS
BEGIN
-- Insert query results into @DailyTripStats
INSERT INTO @DailyTripStats 
SELECT
    -- Cast StartDate as a date
	CAST(StartDate AS date),
    COUNT(ID),
    AVG(Duration)
FROM CapitalBikeShare
WHERE
	DATEPART(month, StartDate) = @Month AND
    DATEPART(year, StartDate) = @Year
-- Group by StartDate as a date
GROUP BY CAST(StartDate AS date)
-- Return
RETURN
END;



-- Execute scalar with select
-- Create @BeginDate
DECLARE @BeginDate AS date = '3/1/2018'
-- Create @EndDate
DECLARE @EndDate AS date = '3/10/2018' 
SELECT
  -- Select @BeginDate
  @BeginDate AS BeginDate,
  -- Select @EndDate
  @EndDate AS EndDate,
  -- Execute SumRideHrsDateRange()
 dbo.SumRideHrsDateRange(@BeginDate, @EndDate) AS TotalRideHrs
 
 
 
 
 -- EXEC scalar
 -- Create @RideHrs
DECLARE @RideHrs AS numeric
-- Execute SumRideHrsSingleDay() and store the result in @RideHrs
EXEC @RideHrs = dbo.SumRideHrsSingleDay @DateParm = '3/5/2018' 
SELECT 
  'Total Ride Hours for 3/5/2018:', 
  @RideHrs
  
  
  -- Execute TVF into variable
  -- Create @StationStats
DECLARE @StationStats TABLE(
	StartStation nvarchar(100), 
	RideCount int, 
	TotalDuration numeric)
-- Populate @StationStats with the results of the function
INSERT INTO @StationStats
SELECT TOP 10 *
-- Execute SumStationStats with 3/15/2018
FROM dbo.SumStationStats('3/15/2018') 
ORDER BY RideCount DESC
-- Select all the records from @StationStats
SELECT * 
FROM @StationStats


-- CREATE OR ALTER
-- Update SumStationStats
CREATE OR ALTER FUNCTION dbo.SumStationStats(@EndDate AS date)
-- Enable SCHEMABINDING
RETURNS TABLE WITH SCHEMABINDING
AS
RETURN
SELECT
	StartStation,
    COUNT(ID) AS RideCount,
    SUM(DURATION) AS TotalDuration
FROM dbo.CapitalBikeShare
-- Cast EndDate as date and compare to @EndDate
WHERE CAST(EndDate AS date) = @Enddate
GROUP BY StartStation;
 
 
 
 
 
 -- CHAPTER 3: Stored Procedures
 -- CREATE PROCEDURE with OUTPUT
 -- Create the stored procedure
CREATE PROCEDURE dbo.cuspSumRideHrsSingleDay
    -- Declare the input parameter
	@DateParm date,
    -- Declare the output parameter
	@RideHrsOut numeric OUTPUT
AS
-- Don't send the row count 
SET NOCOUNT ON
BEGIN
-- Assign the query result to @RideHrsOut
SELECT
	@RideHrsOut = SUM(DATEDIFF(second, StartDate, EndDate))/3600
FROM CapitalBikeShare
-- Cast StartDate as date and compare with @DateParm
WHERE CAST(StartDate AS date) = @DateParm
RETURN
END;



-- Use SP to INSERT
-- Create the stored procedure
CREATE PROCEDURE dbo.cusp_RideSummaryCreate (@DateParm date)
AS
BEGIN
SET NOCOUNT ON
-- Insert into the Date and RideHours columns
INSERT INTO dbo.SumRideHrsSingleDay(Date, RideHours)
-- Insert @DateParm and the result of SumRideHrsSingleDay
VALUES(@DateParm, dbo.SumRideHrsSingleDay(@DateParm)) 

-- Select the record that was just inserted
SELECT
    -- Select Date column
	Date,
    -- Select RideHours column
    RideHours
FROM dbo.RideSummary
-- Check whether Date equals @DateParm
WHERE Date = @DateParm
END;



-- Use SP to UPDATE
-- Create the stored procedure
CREATE PROCEDURE dbo.cuspRideSummaryUpdate
	-- Specify @Date input parameter
	(@Date as date,
     -- Specify @RideHrs input parameter
     @RideHrs numeric(18,0))
AS
BEGIN
SET NOCOUNT ON
-- Update RideSummary
Update RideSummary
-- Set
SET
	Date = @Date,
    RideHours = @RideHrs
-- Include records where Date equals @Date
WHERE Date = @Date
END;


-- Use SP to DELETE
-- Create the stored procedure
CREATE PROCEDURE dbo.cuspRideSummaryDelete
	-- Specify @DateParm input parameter
	(@DateParm Date,
     -- Specify @RowCountOut output parameter
     @RowCountOut int OUTPUT)
AS
BEGIN
-- Delete record(s) where Date equals @DateParm
DELETE FROM dbo.RideSummary
WHERE Date = @DateParm
-- Set @RowCountOut to @@ROWCOUNT
SET  @RowCountOut = @@ROWCOUNT
END;
 
 
 
 
 
 

-- EXECUTE with OUTPUT parameter
-- Create @RideHrs
DECLARE @RideHrs AS numeric(18,0)
-- Execute the stored procedure
EXEC dbo.cuspSumRideHrsSingleDay
    -- Pass the input parameter
	@DateParm = '3/1/2018',
    -- Store the output in RideHrsOut
	@RideHrsOut = @RideHrs OUTPUT
-- Select RideHrs
SELECT @RideHrs AS RideHours;




-- EXECUTE with return value
-- Create @ReturnStatus
DECLARE @ReturnStatus AS int
-- Execute the SP, storing the result in @ReturnStatus
EXEC @ReturnStatus = dbo.cuspRideSummaryUpdate
    -- Specify @DateParm
	@DateParm = '3/1/2018',
    -- Specify @RideHrs
	@RideHrs = 300

-- Select the columns of interest
SELECT
	@ReturnStatus AS ReturnStatus,
    Date,
    RideHours
FROM dbo.ridesummary 
WHERE Date = '3/1/2018';



-- EXECUTE with OUTPUT & return value
-- Create @ReturnStatus
DECLARE @ReturnStatus AS int
-- Create @RowCount
DECLARE @RowCount AS int

-- Execute the SP, storing the result in @ReturnStatus
EXEC @ReturnStatus = dbo.cuspRideSummaryDelete 
    -- Specify @DateParm
	@DateParm = '3/1/2018',
    -- Specify RowCountOut
	@RowCountOut = @RowCount OUTPUT

-- Select the columns of interest
SELECT
	@ReturnStatus AS ReturnStatus,
    @RowCount AS 'RowCount';
    



-- Your very own TRY..CATCH
-- Alter the stored procedure
CREATE OR ALTER PROCEDURE dbo.cuspRideSummaryDelete
	-- (Incorrectly) specify @DateParm
	@DateParm nvarchar(30),
    -- Specify @Error
	@Error nvarchar(max) = NULL OUTPUT
AS
SET NOCOUNT ON
BEGIN
  -- Start of the TRY block
  BEGIN TRY
  	  -- Delete
      DELETE FROM RideSummary
      WHERE Date = @DateParm
  -- End of the TRY block
  END TRY
  -- Start of the CATCH block
  BEGIN CATCH
		SET @Error = 
		'Error_Number: '+ CAST(ERROR_NUMBER() AS VARCHAR) +
		'Error_Severity: '+ CAST(ERROR_SEVERITY() AS VARCHAR) +
		'Error_State: ' + CAST(ERROR_STATE() AS VARCHAR) + 
		'Error_Message: ' + ERROR_MESSAGE() + 
		'Error_Line: ' + CAST(ERROR_LINE() AS VARCHAR)
  -- End of the CATCH block
  END CATCH
END;


-- CATCH an error
-- Create @ReturnCode
DECLARE @ReturnCode int
-- Create @ErrorOut
DECLARE @ErrorOut nvarchar(max)
-- Execute the SP, storing the result in @ReturnCode
execute @ReturnCode = dbo.cuspRideSummaryDelete
    -- Specify @DateParm
	@DateParm = '1/32/2018',
    -- Assign @ErrorOut to @Error
	@Error = @ErrorOut OUTPUT
-- Select @ReturnCode and @ErrorOut
SELECT
	@ReturnCode AS ReturnCode,
    @ErrorOut AS ErrorMessage;
    
    
    
    
    
-- CHAPTER 4: NYC Taxi Ride Case Study
