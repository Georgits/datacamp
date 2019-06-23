use 08_sql_for_exploratory_data_analysis;

-- CHAPTER 1: What's in the database?
SELECT COUNT(*) FROM stackoverflow;
SELECT * FROM stackoverflow;

-- Count missing values
-- Select the count of ticker, 
-- subtract from the total number of rows, 
-- and alias as missing
SELECT count(*) - count(ticker) AS missing
  FROM fortune500;
  
  -- Select the count of profits_change, 
-- subtract from total number of rows, and alias as missing
SELECT count(*) - count(profits_change) AS missing
  FROM fortune500;
  
  -- Select the count of industry, 
-- subtract from total number of rows, and alias as missing
SELECT count(*) - count(industry) AS missing
  FROM fortune500;
  
  
  -- Join tables
  SELECT company.name
-- Table(s) to select from
  FROM company
       INNER JOIN fortune500
       ON fortune500.ticker=company.ticker;