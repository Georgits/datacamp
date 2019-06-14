use 06_reporting_in_sql;


-- CHAPTER 1: Exploring the Olympics Dataset
-- Building the base report
-- Query the sport and distinct number of athletes
SELECT 
	sport, 
    COUNT(DISTINCT(athlete_id)) AS athletes
FROM summer_games
GROUP BY sport
-- Only include the 3 sports with the most athletes
ORDER BY athletes
LIMIT 3;


-- Athletes vs events by sport
-- Query sport, events, and athletes from summer_games
SELECT 
	sport, 
    COUNT(DISTINCT(event)) AS events, 
    COUNT(DISTINCT(athlete_id)) AS athletes
FROM summer_games
GROUP BY sport;


-- Age of oldest athlete by region
-- Select the age of the oldest athlete for each region
SELECT 
	region, 
    MAX(age) AS age_of_oldest_athlete
FROM athletes AS a
-- First JOIN statement
JOIN summer_games AS s
ON a.id = s.athlete_id
-- Second JOIN statement
JOIN countries AS c
ON s.country_id = c.id
GROUP BY region;


-- Number of events in each sport
-- Select sport and events for summer sports
SELECT 
	sport, 
    COUNT(DISTINCT event) AS events
FROM summer_games
GROUP BY sport
UNION
-- Select sport and events for winter sports
SELECT 
	sport, 
    COUNT(DISTINCT event) AS events
FROM winter_games
GROUP BY sport
-- Show the most events at the top of the report
ORDER BY events;


-- Exploring summer_games
-- Update query to explore the unique bronze field values
SELECT DISTINCT medal
FROM summer_games;

-- Add the rows column to your query
SELECT 
	medal, 
	COUNT(*) AS row_num
FROM summer_games
GROUP BY medal;


-- Validating our query
SELECT COUNT(medal) AS total_medals
FROM summer_games;

-- Setup a query that shows bronze_medal by country
SELECT 
	country, 
    COUNT(medal) AS medals
FROM summer_games AS s
JOIN countries AS c
ON s.country_id = c.id
GROUP BY country;


-- Select the total bronze_medals from your query
SELECT SUM(medals)
FROM (
-- Previous query is shown below.  Alias this AS subquery
  SELECT 
      country, 
      COUNT(medal) AS medals
  FROM summer_games AS s
  JOIN countries AS c
  ON s.country_id = c.id
  GROUP BY country) AS subquery
;


-- Report 1: Most decorated summer athletes
-- Pull athlete_name and gold_medals for summer games
SELECT 
	a.name AS athlete_name, 
    COUNT(medal) AS medals
FROM summer_games AS s
JOIN athletes AS a
ON s.athlete_id = a.id
GROUP BY athlete_name
-- Filter for only athletes with 3 gold medals or more
HAVING COUNT(medal) > 2
-- Sort to show the most gold medals at the top
ORDER BY medals DESC;





-- CHAPTER 2: Creating Reports
-- Planning the filter
-- Pull distinct event names found in winter_games
SELECT DISTINCT event
FROM winter_games;


-- JOIN then UNION query
-- Query season, country, and events for all summer events
SELECT 
	'summer' AS season, 
    country, 
    COUNT(DISTINCT event) AS events
FROM summer_games AS s
JOIN countries AS c
ON s.country_id = c.id
GROUP BY country
-- Combine the queries
UNION ALL
-- Query season, country, and events for all winter events
SELECT 
	'winter' AS season, 
    country, 
    COUNT(DISTINCT event) AS events
FROM winter_games AS w
JOIN countries AS c
ON w.country_id = c.id
GROUP BY country
-- Sort the results to show most events at the top
ORDER BY events DESC;



-- UNION then JOIN query
-- Add outer layer to pull season, country and unique events
SELECT 
	season, 
    country, 
    COUNT(DISTINCT event) AS events
FROM
    -- Pull season, country_id, and event for both seasons
    (SELECT 
     	'summer' AS season, 
     	country_id, 
     	event
    FROM summer_games
    UNION ALL
    SELECT 
     	'winter' AS season, 
     	country_id, 
     	event
    FROM winter_games) AS subquery
JOIN countries AS c
ON subquery.country_id = c.id
-- Group by any unaggregated fields
GROUP BY season, country
-- Order to show most events at the top
ORDER BY events DESC;



-- CASE statement refresher
SELECT 
	name,
    -- Output 'Tall Female', 'Tall Male', or 'Other'
	CASE WHEN gender = 'F' AND height >= 175 THEN 'Tall Female'
    WHEN gender = 'M' AND height >= 190 THEN 'Tall Male'
    ELSE 'Other' END AS segment
FROM athletes;



-- BMI bucket by sport
-- Pull in sport, bmi_bucket, and athletes
SELECT 
	sport,
    -- Bucket BMI in three groups: <.25, .25-.30, and >.30	
    CASE WHEN (100 * weight / (height * height)) < .25 THEN '<.25'
    WHEN (100 * weight / (height * height)) <= .30 THEN '.25-.30'
    WHEN (100 * weight / (height * height)) > .30 THEN '>.30' END AS bmi_bucket,
    COUNT(DISTINCT name) AS athletes
FROM summer_games AS s
JOIN athletes AS a
ON s.athlete_id = a.id
-- GROUP BY non-aggregated fields
GROUP BY bmi_bucket, sport
-- Sort by sport and then by athletes in descending order
ORDER BY sport, athletes DESC;


-- Troubleshooting CASE statements
/*-- Query from last exercise shown below.  Comment it out.
SELECT 
	sport,
    CASE WHEN weight/height^2*100 <.25 THEN '<.25'
    WHEN weight/height^2*100 <=.30 THEN '.25-.30'
    WHEN weight/height^2*100 >.30 THEN '>.30' END AS bmi_bucket,
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games AS s
JOIN athletes AS a
ON s.athlete_id = a.id
GROUP BY sport, bmi_bucket
ORDER BY sport, athletes DESC; */

-- Show height, weight, and bmi for all athletes
SELECT 
	height,
    weight,
    weight/height^2*100 AS bmi
FROM athletes
-- Filter for NULL bmi values
WHERE weight/height^2*100 is null;


-- Uncomment the original query
SELECT 
	sport,
    CASE WHEN weight/(height*height)*100 <.25 THEN '<.25'
    WHEN weight/(height*height) *100 <=.30 THEN '.25-.30'
    WHEN weight/(height*height)*100 >.30 THEN '>.30'
    -- Add ELSE statement to output 'no weight recorded'
    ELSE 'no weight recorded' END AS bmi_bucket,
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games AS s
JOIN athletes AS a
ON s.athlete_id = a.id
GROUP BY sport, bmi_bucket
ORDER BY sport, athletes DESC;



-- Filtering with a JOIN
-- Pull summer bronze_medals, silver_medals, and gold_medals
SELECT 
	SUM(bronze) AS bronze_medals, 
    SUM(silver) AS silver_medals, 
    SUM(gold) AS gold_medals
FROM 
(
SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    athlete_id
	FROM summer_games) AS s
JOIN athletes AS a
ON s.athlete_id = a.id
-- Filter for athletes age 16 or below
WHERE age < 17;



-- Filtering with a subquery
-- Pull summer bronze_medals, silver_medals, and gold_medals
SELECT 
    SUM(bronze) AS bronze_medals, 
    SUM(silver) AS silver_medals, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    athlete_id
	FROM summer_games) AS s
-- Add the WHERE statement below
WHERE s.athlete_id IN
    -- Create subquery list for athlete_ids age 16 or below    
    (SELECT id
     FROM athletes
     WHERE age < 17);


