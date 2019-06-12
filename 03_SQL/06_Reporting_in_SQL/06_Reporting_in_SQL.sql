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