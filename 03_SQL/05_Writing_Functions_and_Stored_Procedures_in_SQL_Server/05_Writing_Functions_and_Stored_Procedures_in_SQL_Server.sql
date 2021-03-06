use 05_writing_functions_and_stored_procedures_in_sql_server;

-- Das ist ein T-SQL Script und funktionniert ohne Anpassung in MySQL nicht
-- Wie das geht, s. beispielhaft direkt in nachfolgenden Zeilen

-- DATASETS
-- https://s3.amazonaws.com/capitalbikeshare-data/index.html
-- https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page


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
-- Use EDA to find impossible scenarios
SELECT
	-- PickupDate is after today
	COUNT (CASE WHEN PickupDate > GetDate() THEN 1 END) AS 'FuturePickup',
    -- DropOffDate is after today
	COUNT (CASE WHEN DropoffDate > GetDate() THEN 1 END) AS 'FutureDropOff',
    -- PickupDate is after DropOffDate
	COUNT (CASE WHEN PickupDate > DropOffDate THEN 1 END) AS 'PickupBeforeDropoff',
    -- TripDistance is 0
	COUNT (CASE WHEN TripDistance = 0 THEN 1 END) AS 'ZeroTripDistance'  
FROM YellowTripData;



-- Mean imputation
-- Create the stored procedure
CREATE PROCEDURE dbo.cuspImputeTripDistanceMean
AS
BEGIN
-- Specify @AvgTripDistance variable
DECLARE @AvgTripDistance AS numeric (18,4)

-- Calculate the average trip distance
SELECT @AvgTripDistance = AVG(TripDistance) 
FROM YellowTripData
-- Only include trip distances greater than 0
WHERE TripDistance > 0

-- Update the records where trip distance is 0
UPDATE YellowTripData
SET TripDistance =  @AvgTripDistance
WHERE TripDistance = 0
END;




-- Hot Deck imputation
-- Create the function
CREATE FUNCTION dbo.GetTripDistanceHotDeck()
-- Specify return data type
RETURNS numeric(18,4)
AS 
BEGIN
RETURN
	-- Select the first TripDistance value
	(SELECT TOP 1 TripDistance
	FROM YellowTripData
    -- Sample 1000 records
	TABLESAMPLE(1000 rows)
    -- Only include records where TripDistance is > 0
	WHERE TripDistance > 0)
END;


-- CREATE FUNCTIONs
-- Create the function
CREATE FUNCTION dbo.ConvertMileToKm (@Miles numeric(18,2))
-- Specify return data type
RETURNS numeric(18,2)
AS
BEGIN
RETURN
	-- Convert Miles to Kilometers
	(SELECT @Miles * 1.609)
END;

-- Create the function
CREATE FUNCTION dbo.ConvertDollar
	-- Specify @DollarAmt parameter
	( @DollarAmt numeric(18,2),
     -- Specify @ExchangeRate parameter
     @ExchangeRate numeric(18,2))
-- Specify return data type
RETURNS numeric(18,2)
AS
BEGIN
RETURN
	-- Multiply @ExchangeRate and @DollarAmt
	(SELECT @ExchangeRate * @DollarAmt)
END;

-- Create the function
CREATE FUNCTION dbo.GetShiftNumber (@Hour integer)
-- Specify return data type
RETURNS int
AS
BEGIN
RETURN
	-- 12am (0) to 9am (9) shift
	(CASE WHEN @Hour >= 0 AND @Hour < 9 THEN 1
     	  -- 9am (9) to 5pm (17) shift
		 WHEN @Hour >= 9 AND @Hour < 17 THEN 2
          -- 5pm (17) to 12am (24) shift
	     WHEN @Hour >= 17 AND @Hour < 24 THEN 3 END)
END;


-- Test FUNCTIONs
SELECT
	-- Select the first 100 records of PickupDate
	TOP 100 PickupDate,
    -- Determine the shift value of PickupDate
	dbo.GetShiftNumber(DATENAME(hour, PickupDate)) AS 'Shift',
    -- Select FareAmount
	FareAmount,
    -- Convert FareAmount to Euro
	dbo.ConvertDollar(FareAmount, 0.87) AS 'FareinEuro',
    -- Select TripDistance
	TripDistance,
    -- Convert TripDistance to kilometers
	dbo.ConvertMiletoKm(TripDistance) AS 'TripDistanceinKM'
FROM YellowTripData
-- Only include records for the 2nd shift
WHERE dbo.GetShiftNumber(DATENAME(hour, PickupDate)) = 2;


-- Logical weekdays with Hot Deck
SELECT
    -- Select the pickup day of week
	DATENAME(weekday, PickupDate) as DayofWeek,
    -- Calculate TotalAmount per TripDistance
	CAST(AVG(TotalAmount/
            -- Select TripDistance if it's more than 0
			CASE WHEN TripDistance > 0 THEN TripDistance
                 -- Use GetTripDistanceHotDeck()
     			 ELSE dbo.GetTripDistanceHotDeck() END) as decimal(10,2)) as 'AvgFare'
FROM YellowTripData
GROUP BY DATENAME(weekday, PickupDate)
-- Order by the PickupDate day of week
ORDER BY
     CASE WHEN DATENAME(weekday, PickupDate) = 'Monday' THEN 1
         WHEN DATENAME(weekday, PickupDate) = 'Tuesday' THEN 2
         WHEN DATENAME(weekday, PickupDate) = 'Wednesday' THEN 3
         WHEN DATENAME(weekday, PickupDate) = 'Thursday' THEN 4
         WHEN DATENAME(weekday, PickupDate) = 'Friday' THEN 5
         WHEN DATENAME(weekday, PickupDate) = 'Saturday' THEN 6
         WHEN DATENAME(weekday, PickupDate) = 'Sunday' THEN 7
END ASC;



-- Format for Germany
SELECT
    -- Cast PickupDate as a date and display as a German date
	FORMAT(CAST(PickupDate AS date), 'd', 'de-de') AS 'PickupDate',
	Zone.Borough,
    -- Display TotalDistance in the German format
	FORMAT(SUM(TripDistance), 'n', 'de-de') AS 'TotalDistance',
    -- Display TotalRideTime in the German format
	FORMAT(SUM(DATEDIFF(minute, PickupDate, DropoffDate)), 'n', 'de-de') AS 'TotalRideTime',
    -- Display TotalFare in German currency
	FORMAT(SUM(TotalAmount), 'c', 'de-de') AS 'TotalFare'
FROM YellowTripData
INNER JOIN TaxiZoneLookup AS Zone 
ON PULocationID = Zone.LocationID 
GROUP BY
	CAST(PickupDate as date),
    Zone.Borough 
ORDER BY
	CAST(PickupDate as date),
    Zone.Borough;
    
    
    
    
-- NYC Borough statistics SP
CREATE OR ALTER PROCEDURE dbo.cuspBoroughRideStats
AS
BEGIN
SELECT
    -- Calculate the pickup weekday
	DATENAME(weekday, PickupDate) AS 'Weekday',
    -- Select the Borough
	Zone.Borough AS 'PickupBorough',
    -- Display AvgFarePerKM as German currency
	FORMAT(AVG(dbo.ConvertDollar(TotalAmount, .88)/dbo.ConvertMiletoKM(TripDistance)), 'c', 'de-de') AS 'AvgFarePerKM',
    -- Display RideCount in the German format
	FORMAT(COUNT(ID), 'n', 'de-de') AS 'RideCount',
    -- Display TotalRideMin in the German format
	FORMAT(SUM(DATEDIFF(SECOND, PickupDate, DropOffDate))/60, 'n', 'de-de') AS 'TotalRideMin'
FROM YellowTripData
INNER JOIN TaxiZoneLookup AS Zone 
ON PULocationID = Zone.LocationID
-- Only include records where TripDistance is greater than 0
WHERE TripDistance > 0
-- Group by pickup weekday and Borough
GROUP BY DATENAME(WEEKDAY, PickupDate), Zone.Borough
ORDER BY CASE WHEN DATENAME(WEEKDAY, PickupDate) = 'Monday' THEN 1
	     	  WHEN DATENAME(WEEKDAY, PickupDate) = 'Tuesday' THEN 2
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Wednesday' THEN 3
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Thursday' THEN 4
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Friday' THEN 5
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Saturday' THEN 6
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Sunday' THEN 7 END,  
		 SUM(DATEDIFF(SECOND, PickupDate, DropOffDate))/60
DESC
END;



-- NYC Borough statistics results
-- Create SPResults
DECLARE @SPResults TABLE(
  	-- Create Weekday
	Weekday nvarchar(30),
    -- Create Borough
	Borough nvarchar(30),
    -- Create AvgFarePerKM
	AvgFarePerKM nvarchar(30),
    -- Create RideCount
	RideCount	nvarchar(30),
    -- Create TotalRideMin
	TotalRideMin	nvarchar(30))

-- Insert the results into @SPResults
INSERT INTO @SPResults
-- Execute the SP
EXEC dbo.cuspBoroughRideStats

-- Select all the records from @SPresults 
SELECT * 
FROM @SPresults;





-- Pickup locations by shift
-- Create the stored procedure
CREATE PROCEDURE dbo.cuspPickupZoneShiftStats
	-- Specify @Borough parameter
	@Borough nvarchar(30)
AS
BEGIN
SELECT
	DATENAME(WEEKDAY, PickupDate) as 'Weekday',
    -- Calculate the shift number
	dbo.GetShiftNumber(datepart(hour, PickupDate)) as 'Shift',
	Zone.Zone as 'Zone',
	FORMAT(AVG(dbo.ConvertDollar(TotalAmount, .77)/dbo.ConvertMiletoKM(TripDistance)), 'c', 'de-de') AS 'AvgFarePerKM',
	FORMAT(COUNT (ID),'n', 'de-de') as 'RideCount',
	FORMAT(SUM(DATEDIFF(SECOND, PickupDate, DropOffDate))/60, 'n', 'de-de') as 'TotalRideMin'
FROM YellowTripData
INNER JOIN TaxiZoneLookup as Zone on PULocationID = Zone.LocationID 
WHERE
	dbo.ConvertMiletoKM(TripDistance) > 0 AND
	Zone.Borough = @Borough
GROUP BY
	DATENAME(WEEKDAY, PickupDate),
    -- Group by shift
	dbo.GetShiftNumber(datepart(hour, PickupDate)),  
	Zone.Zone
ORDER BY CASE WHEN DATENAME(WEEKDAY, PickupDate) = 'Monday' THEN 1
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Tuesday' THEN 2
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Wednesday' THEN 3
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Thursday' THEN 4
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Friday' THEN 5
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Saturday' THEN 6
              WHEN DATENAME(WEEKDAY, PickupDate) = 'Sunday' THEN 7 END,
         -- Order by shift
         dbo.GetShiftNumber(datepart(hour, PickupDate)),
         SUM(DATEDIFF(SECOND, PickupDate, DropOffDate))/60 DESC
END;


-- Pickup locations by shift results
-- Create @Borough
DECLARE @Borough AS nvarchar(30) = 'Manhattan'
-- Execute the SP
EXEC dbo.cuspPickupZoneShiftStats
    -- Pass @Borough
    -- When executing the SP, assign @Borough variable to @Borough input parameter.
	@Borough = @Borough;
