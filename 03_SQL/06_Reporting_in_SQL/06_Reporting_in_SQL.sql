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


-- Report 2: Top athletes in nobel-prized countries
-- Pull event and unique athletes from summer_games 
SELECT 	
	event,
	COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
GROUP BY event;

-- Pull event and unique athletes from summer_games 
SELECT 
	event, 
    -- Add the gender field below
    CASE WHEN event LIKE '%Women%' THEN 'female'
    ELSE 'male' END AS gender,
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
GROUP BY event;

-- Pull event and unique athletes from summer_games 
SELECT 
    event,
    -- Add the gender field below
    CASE WHEN event LIKE '%Women%' THEN 'female' 
    ELSE 'male' END AS gender,
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
-- Only include countries that won a nobel prize
WHERE country_id IN 
	(SELECT country_id
    FROM country_stats
    WHERE nobel_prize_winners > 0)
GROUP BY event;


-- Pull event and unique athletes from summer_games 
SELECT 
    event,
    -- Add the gender field below
    CASE WHEN event LIKE '%Women%' THEN 'female' 
    ELSE 'male' END AS gender,
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
-- Only include countries that won a nobel prize
WHERE country_id IN 
	(SELECT country_id 
    FROM country_stats 
    WHERE nobel_prize_winners > 0)
GROUP BY event
-- Add the second query below and combine with a UNION
UNION
SELECT 
    event,
    -- Add the gender field below
    CASE WHEN event LIKE '%Women%' THEN 'female' 
    ELSE 'male' END AS gender,
    COUNT(DISTINCT athlete_id) AS athletes
FROM winter_games
-- Only include countries that won a nobel prize
WHERE country_id IN 
	(SELECT country_id 
    FROM country_stats 
    WHERE nobel_prize_winners > 0)
GROUP BY event
-- Order and limit the final output
ORDER BY athletes DESC
LIMIT 10;



-- CHAPTER 3: Cleaning & Validation
-- Identifying data types
-- Pull column_name & data_type from the columns table
SELECT 
	column_name,
    data_type
FROM information_schema.columns
-- Filter for the table 'country_stats'
WHERE table_name = 'country_stats';


-- Interpreting error messages
-- Comment out the previous query
-- Comment out the previous query
SELECT AVG(CAST(population AS DECIMAL)) AS avg_population
FROM country_stats;

-- Using date functions on strings
SELECT 
	year,
    -- Pull decade, decade_truncate, and the world's gdp
   /* POSTGRESSQL
   DATEPART('decade', CAST(year AS date)) AS decade,
    DATE_TRUNC('decade', CAST(year AS date)) AS decade_truncated, */
    SUM(gdp) AS world_gdp
FROM country_stats
-- Group and order by year in descending order
GROUP BY year
ORDER BY year DESC;


-- String functions
-- Convert country to lower case
SELECT 
	country, 
    LOWER(country) AS country_altered
FROM countries
GROUP BY country;

/* POSTGRESSQL
-- Convert country to proper case
SELECT 
	country, 
    INITCAP(country) AS country_altered
FROM countries
GROUP BY country;
*/

-- Output the left 3 characters of country
SELECT 
	country, 
    LEFT(country, 3) AS country_altered
FROM countries
GROUP BY country;


-- Output all characters starting with position 7
SELECT 
	country, 
    SUBSTRING(country from 7) AS country_altered
FROM countries
GROUP BY country;


-- Replacing and removing substrings
SELECT 
	region, 
    -- Replace all '&' characters with the string 'and'
    REPLACE(region, '&', 'and') AS character_swap,
    -- Remove all periods
    REPLACE(region, '.', '') AS character_remove
FROM countries
WHERE region LIKE '%Latin%'
GROUP BY region;

SELECT
	region,
	COUNT(*)
 from countries
WHERE region LIKE '%Latin%'
 GROUP BY region;
 
 
 SELECT 
	region, 
    -- Replace all '&' characters with the string 'and'
    REPLACE(region,'&','and') AS character_swap,
    -- Remove all periods
    REPLACE(region,'.','') AS character_remove,
    -- Combine the functions to run both changes at once
    REPLACE(REPLACE(region,'&','and'), '.','') AS character_swap_and_remove
FROM countries
WHERE region LIKE '%Latin%'
GROUP BY region;



-- Fixing incorrect groupings
-- Pull event and unique athletes from summer_games_messy 
SELECT
    -- Remove trailing spaces and alias as event_fixed
	event, 
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
-- Update the group by accordingly
GROUP BY event;


-- Pull event and unique athletes from summer_games_messy 
SELECT 
    -- Remove dashes from all event values
    TRIM(event) AS event_fixed, 
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
-- Update the group by accordingly
GROUP BY event_fixed;


-- Pull event and unique athletes from summer_games_messy 
SELECT 
    -- Remove dashes from all event values
    REPLACE(TRIM(event), '-','') AS event_fixed, 
    COUNT(DISTINCT athlete_id) AS athletes
FROM summer_games
-- Update the group by accordingly
GROUP BY event_fixed;




-- Filtering out nulls
-- Show total gold_medals by country
SELECT 
	country,
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id
    FROM winter_games) AS w
JOIN countries AS c
ON w.country_id = c.id
GROUP BY country
-- Order by gold_medals in descending order
ORDER BY gold_medals DESC;



-- Show total gold_medals by country
SELECT 
	country, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id
    FROM winter_games) AS w
JOIN countries AS c
ON w.country_id = c.id
-- Removes any row with no gold medals
WHERE gold IS NOT NULL
GROUP BY country
-- Order by gold_medals in descending order
ORDER BY gold_medals DESC;


-- Show total gold_medals by country
SELECT 
	country, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id
    FROM winter_games) AS w
JOIN countries AS c
ON w.country_id = c.id
-- Comment out the WHERE statement
-- WHERE gold IS NOT NULL
GROUP BY country
-- Replace WHERE statement with equivalent HAVING statement
HAVING SUM(gold) IS NOT NULL
-- Order by gold_medals in descending order
ORDER BY gold_medals DESC;



-- Fixing calculations with coalesce
-- Pull events and golds by athlete_id for summer events
SELECT 
    athlete_id,
    COUNT(event) AS total_events, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    event,
    athlete_id
    FROM summer_games) AS s
GROUP BY athlete_id
-- Order by total_events descending and athlete_id ascending
ORDER BY total_events DESC, athlete_id ASC;


-- Pull events and golds by athlete_id for summer events
SELECT 
    athlete_id,
    -- Add a field that averages the existing gold field
    AVG(gold) AS avg_golds,
    COUNT(event) AS total_events, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    event,
    athlete_id
    FROM summer_games) AS s
GROUP BY athlete_id
-- Order by total_events descending and athlete_id ascending
ORDER BY total_events DESC, athlete_id ASC;



-- Pull events and golds by athlete_id for summer events
SELECT 
    athlete_id,
    -- Add a field that averages the existing gold field
    AVG(COALESCE(gold,0)) AS avg_golds,
    COUNT(event) AS total_events, 
    SUM(gold) AS gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    event,
    athlete_id
    FROM summer_games) AS s
GROUP BY athlete_id
-- Order by total_events descending and athlete_id ascending
ORDER BY total_events DESC, athlete_id ASC;




-- Identifying duplication
-- Pull total gold_medals for winter sports
SELECT sum(gold) as gold_medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold
    FROM winter_games) AS w;
    
    
-- Show gold_medals and avg_gdp by country_id
SELECT 
	w.country_id, 
    SUM(gold) AS gold_medals, 
    AVG(gdp) AS avg_gdp
FROM 
(SELECT
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id
    FROM winter_games) AS w
JOIN country_stats AS c
-- Only join on the country_id fields
ON w.country_id = c.country_id
GROUP BY w.country_id;



-- Calculate the total gold_medals in your query
SELECT SUM(gold_medals)
FROM
	(SELECT 
	w.country_id, 
    SUM(gold) AS gold_medals, 
    AVG(gdp) AS avg_gdp
		FROM 
		(SELECT
			CASE WHEN medal = 'Gold' THEN 1
			ELSE NULL END AS gold,
			country_id
			FROM winter_games) AS w
		JOIN country_stats AS c
		-- Only join on the country_id fields
		ON w.country_id = c.country_id
		GROUP BY w.country_id) AS subquery;
        
        
-- Fixing duplication through a JOIN
SELECT SUM(gold_medals)
FROM
	(SELECT 
	w.country_id, 
    SUM(gold) AS gold_medals, 
    AVG(gdp) AS avg_gdp
		FROM 
		(SELECT
			CASE WHEN medal = 'Gold' THEN 1
			ELSE NULL END AS gold,
			country_id,
            year
			FROM winter_games) AS w
		JOIN country_stats AS c
		-- Only join on the country_id fields
		ON w.country_id = c.country_id AND CAST(w.year AS date) = CAST(c.year AS date)
		GROUP BY w.country_id) AS subquery;
        
        
        
-- Report 3: Countries with high medal rates
SELECT 
	c.country,
    -- Add the three medal fields using one sum function
	SUM(COALESCE(bronze,0) + COALESCE(silver,0) + COALESCE(gold,0)) AS medals
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id
    FROM summer_games) AS s
JOIN countries AS c
ON s.country_id = c.id
GROUP BY country
ORDER BY medals DESC;



SELECT 
	c.country,
    -- Pull in pop_in_millions and medals_per_million 
	population,
    -- Add the three medal fields using one sum function
	SUM(COALESCE(bronze,0) + COALESCE(silver,0) + COALESCE(gold,0)) AS medals,
	SUM(COALESCE(bronze,0) + COALESCE(silver,0) + COALESCE(gold,0)) / CAST(cs.population AS decimal)*1000000 AS medals_per_million
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id,
    year
    FROM summer_games) AS s
JOIN countries AS c 
ON s.country_id = c.id
-- Update the newest join statement to remove duplication
JOIN country_stats AS cs 
ON s.country_id = cs.country_id AND CAST(s.year AS date) = CAST(cs.year AS date)
GROUP BY c.country, population
ORDER BY medals DESC;



SELECT 
	-- Clean the country field to only show country_code
    LEFT(REPLACE(UPPER(TRIM(c.country)), '.', ''), 3) AS country_code,
    -- Pull in pop_in_millions and medals_per_million 
	population,
    -- Add the three medal fields using one sum function
	SUM(COALESCE(bronze,0) + COALESCE(silver,0) + COALESCE(gold,0)) AS medals,
	SUM(COALESCE(bronze,0) + COALESCE(silver,0) + COALESCE(gold,0)) / CAST(cs.population AS decimal)*1000000 AS medals_per_million
FROM 
(SELECT
	CASE WHEN medal = 'Bronze' THEN 1
	ELSE NULL END AS bronze,
	CASE WHEN medal = 'Silver' THEN 1
	ELSE NULL END AS silver,
	CASE WHEN medal = 'Gold' THEN 1
	ELSE NULL END AS gold,
    country_id,
    year
    FROM summer_games) AS s
JOIN countries AS c 
ON s.country_id = c.id
-- Update the newest join statement to remove duplication
JOIN country_stats AS cs 
ON s.country_id = cs.country_id AND s.year = CAST(cs.year AS date)
-- Filter out null populations
WHERE cs.population IS NOT NULL
GROUP BY c.country, population
-- Keep only the top 25 medals_per_million rows
ORDER BY medals_per_million DESC
LIMIT 25;