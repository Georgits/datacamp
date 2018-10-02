-- Chapter 1. Introduction to joins

-- Onboarding | Tables ----
SELECT cities.name AS city, countries.name AS country, region
FROM cities
INNER JOIN countries
ON cities.country_code = countries.code;


-- Inner join (2) ----
SELECT c.code AS country_code, c.name, e.year, e.inflation_rate
FROM countries AS c
INNER JOIN economies AS e
ON c.code = e.code;



-- Inner join (3) ----
SELECT c.code, name, region, e.year, fertility_rate, unemployment_rate
FROM countries AS c
INNER JOIN populations AS p
ON c.code = p.country_code
INNER JOIN economies AS e
ON c.code = e.code AND p.year = e.year;


-- Inner join with using --
-- When joining tables with a common field nam
SELECT c.name AS country, continent, l.name AS language, official
FROM countries AS c
INNER JOIN languages AS l
USING(code);


-- Self-join --
SELECT p1.country_code,
       p1.size AS size2010,
       p2.size AS size2015,
       ((p2.size - p1.size)/p1.size * 100.0) AS growth_perc
FROM populations AS p1
INNER JOIN populations AS p2
ON p1.country_code = p2.country_code
    AND p1.year = p2.year - 5;
	
	
	
-- Case when and then --
SELECT name, continent, code, surface_area,
        -- first case
    CASE WHEN surface_area > 2000000 THEN 'large'
        -- second case
        WHEN surface_area > 350000 THEN 'medium'
        -- else clause + end
        ELSE 'small' END
        AS geosize_group
INTO countries_plus
FROM countries;



-- Inner challenge --
SELECT country_code, size,
    CASE WHEN size > 50000000 THEN 'large'
        WHEN size > 1000000 THEN 'medium'
        ELSE 'small' END
        AS popsize_group
INTO pop_plus
FROM populations
WHERE year = 2015;

SELECT name, continent, geosize_group, popsize_group
FROM countries_plus AS c
INNER JOIN pop_plus AS p
ON c.code = p.country_code
ORDER BY geosize_group;






-- Chapter 2. Outer joins and cross joins

-- Left Join --
-- get the city name (and alias it), the country code,
-- the country name (and alias it), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- specify left table
FROM cities AS c1
-- specify right table and type of join
INNER JOIN countries AS c2
-- how should the tables be matched?
ON c1.country_code = c2.code
-- sort based on descending country code
ORDER BY code DESC;

-- Change the code to perform a LEFT JOIN instead of an INNER JOIN. After executing this query, note how many records the query result contains.

-- get the city name (and alias it), the country code,
-- the country name (and alias it), the region,
-- and the city proper population
SELECT c1.name AS city, code, c2.name AS country,
       region, city_proper_pop
-- specify left table
FROM cities AS c1
-- specify right table and type of join
LEFT JOIN countries AS c2
-- how should the tables be matched?
ON c1.country_code = c2.code
-- sort based on descending country code
ORDER BY code DESC;


-- Left join (2)
/*
select country name AS country, the country's local name,
the language name AS language, and
the percent of the language spoken in the country
*/
SELECT c.name AS country, local_name, l.name AS language, percent
-- countries on the left (alias as c)
FROM countries AS c
-- appropriate join with languages (as l) on the right
LEFT JOIN languages AS l
-- give fields to match on
ON c.code = l.code
-- sort by descending country name
ORDER BY c.name DESC;


-- Left join (3)
-- select name, region, and gdp_percapita
SELECT c.name, region, gdp_percapita
-- from countries (alias c) on the left
FROM countries AS c
-- left join with economies (alias e)
LEFT JOIN economies AS e
-- match on code fields
ON c.code = e.code
-- focus on 2010 entries
WHERE year = 2010;

-- Select region, average gdp_percapita (alias avg_gdp)
SELECT region, avg(gdp_percapita) AS avg_gdp
-- From countries (alias c) on the left
FROM countries AS c
-- Join with economies (alias e)
LEFT JOIN economies AS e
-- Match on code fields
ON c.code = e.code
-- Focus on 2010 
WHERE year = 2010
-- Group by region
GROUP BY region
-- Order by avg_gdp, descending
ORDER BY avg_gdp DESC;


-- Right join
-- convert this code to use RIGHT JOINs instead of LEFT JOINs
/*
SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM cities
LEFT JOIN countries
ON cities.country_code = countries.code
LEFT JOIN languages
ON countries.code = languages.code
ORDER BY city, language;
*/

SELECT cities.name AS city, urbanarea_pop, countries.name AS country,
       indep_year, languages.name AS language, percent
FROM languages
RIGHT JOIN countries
ON languages.code = countries.code
RIGHT JOIN cities
ON countries.code = cities.country_code
ORDER BY city, language;



-- Full join
SELECT name AS country, code, region, basic_unit
FROM countries
FULL JOIN currencies
USING (code)
WHERE region = 'North America' OR region IS NULL
ORDER BY region;


-- Full join (2)
-- Choose records in which countries.name starts with the capital letter 'V' or is NULL. 
SELECT countries.name, code, languages.name AS language
FROM languages
FULL JOIN countries
USING (code)
WHERE countries.name LIKE 'V%' OR countries.name IS NULL
ORDER BY countries.name;


-- Full join (3)
SELECT c1.name AS country, region, l.name AS language,
       basic_unit, 	frac_unit
FROM countries AS c1
FULL JOIN languages AS l
USING (code)
FULL JOIN currencies AS c2
USING (code)
WHERE region LIKE 'M%esia';


-- A table of two cities
SELECT c.name AS city, l.name AS language
FROM cities AS c        
CROSS JOIN languages  AS l
WHERE c.name LIKE 'Hyder%';

SELECT c.name AS city, l.name AS language
FROM cities AS c        
INNER JOIN languages  AS l
ON c.country_code = l.code
WHERE c.name LIKE 'Hyder%';


-- Outer challenge
SELECT c.name AS country, region, life_expectancy AS life_exp
FROM countries AS c
LEFT JOIN populations AS p
ON c.code = p.	country_code
WHERE year = 2010 
ORDER BY life_exp
LIMIT 5;




-- CHAPTER 3: Set theory clauses
-- Union
-- pick specified columns from 2010 table
SELECT  *
-- 2010 table will be on top
FROM  economies2010
-- which set theory clause?
UNION
-- pick specified columns from 2015 table
SELECT  * 
-- 2015 table on the bottom
FROM  economies2015
-- order accordingly
ORDER BY code, year;


-- Union (2)
SELECT country_code
FROM cities
UNION
SELECT code AS country_code
FROM currencies
ORDER BY country_code;


-- Union all
SELECT code, year
FROM economies
UNION ALL
SELECT country_code, year
FROM populations
ORDER BY code, year;


-- Intersect
SELECT code, year
FROM economies
INTERSECT
SELECT country_code, year
FROM populations
ORDER BY code, year;


-- Intersect (2)
SELECT name
FROM countries
INTERSECT
SELECT name
FROM cities;



-- Except
SELECT name
FROM cities
EXCEPT
SELECT capital
FROM countries
ORDER BY name;



-- Except (2)
SELECT capital
FROM countries
EXCEPT
SELECT name
FROM cities
ORDER bY capital;


-- Semi-join
SELECT DISTINCT name
FROM languages
WHERE code IN
    (SELECT code
    FROM countries
    WHERE region = 'Middle East')
ORDER BY name;


-- Diagnosing problems using anti-join
SELECT COUNT(*)
FROM countries
WHERE continent = 'Oceania';

SELECT c1.code, name, basic_unit AS currency
FROM countries AS c1
INNER JOIN currencies AS c2
ON c1.code = c2.code
WHERE c1.continent = 'Oceania';

SELECT code, name
FROM countries
WHERE continent = 'Oceania'
    AND code NOT IN
    (SELECT code 
     FROM currencies);

	 
	 
-- Set theory challenge
-- select the city name
SELECT name
-- alias the table where city name resides
FROM cities AS c1
-- choose only records matching the result of multiple set theory clauses
WHERE country_code IN
(
    -- select appropriate field from economies AS e
    SELECT e.code
    FROM economies AS e
    -- get all additional (unique) values of the field from currencies AS c2  
    UNION
    SELECT DISTINCT c2.code
    FROM currencies AS c2
    -- exclude those appearing in populations AS p
    EXCEPT
    SELECT p.	country_code
    FROM populations AS p
);



-- CHAPTER 4: Subqueries
-- Subquery inside where
SELECT AVG(life_expectancy)
FROM populations
WHERE year = 2015;

SELECT *
FROM populations
WHERE life_expectancy > 1.15 * (
    SELECT AVG(life_expectancy)
    FROM populations
    WHERE year = 2015)
    AND year = 2015;

	
-- Subquery inside where (2)
-- select the appropriate fields
SELECT name, country_code, urbanarea_pop
-- from the cities table
FROM cities
-- with city name in the field of capital cities
WHERE name IN
  (SELECT capital
   FROM countries)
ORDER BY urbanarea_pop DESC;


-- Subquery inside select
-- Diese zwei Snippets machen das Gleiche

SELECT countries.name AS country, COUNT(*) AS cities_num
FROM cities
INNER JOIN countries
ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;
 
SELECT countries.name AS country,
  (SELECT COUNT(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num DESC, country
LIMIT 9;


-- Subquery inside from
SELECT code, COUNT(*) AS lang_num
FROM languages
GROUP BY code;

SELECT 	local_name, lang_num
FROM countries,
    (SELECT code, COUNT(*) AS lang_num
     FROM languages
     GROUP BY code) AS subquery
WHERE countries.code = subquery.code
ORDER BY lang_num DESC;



-- Advanced subquery
SELECT name, continent, inflation_rate
FROM countries
INNER JOIN economies
USING(code)
WHERE year = 2015;


SELECT max(inflation_rate) AS max_inf
FROM (
    SELECT name, continent, inflation_rate
    FROM countries
    INNER JOIN economies
    USING(code)
    WHERE year = 2015) AS subquery
GROUP BY continent;


SELECT name, continent, inflation_rate
FROM countries
INNER JOIN economies 
ON countries.code = economies.code
WHERE year = 2015
    AND inflation_rate IN (
        SELECT max(inflation_rate) AS max_inf
        FROM (
            SELECT name, continent, inflation_rate
            FROM countries
            INNER JOIN economies
            ON countries.code = economies.code
            WHERE year = 2015) AS subquery
            GROUP BY continent);
			
			
			
-- Subquery challenge
SELECT code, inflation_rate, unemployment_rate
FROM economies
WHERE year = 2015 AND code NOT IN
  (SELECT code
   FROM countries
   WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
ORDER BY inflation_rate;


-- Final challenge
-- In this exercise, you'll need to get the country names and other 2015 data in the economies table and the countries table for Central American countries with an official language.
SELECT DISTINCT c.name, total_investment, imports
FROM countries AS c
LEFT JOIN economies AS e
ON (c.code = e.code
  AND c.code IN (
    SELECT l.code
    FROM languages AS l
    WHERE official = 'true'
  ) )
WHERE year = 2015 AND region = 'Central America'
ORDER BY c.name;



-- Final challenge (2)
--  calculate the average fertility rate for each region in 2015.
-- choose fields
SELECT region, continent, AVG(fertility_rate) AS avg_fert_rate
-- left table
FROM countries AS c
-- right table
INNER JOIN populations AS p
-- join conditions
ON c.code = p.country_code
-- specific records matching a condition
WHERE year = 2015
-- aggregated for each what?
GROUP BY continent, region
-- how should we sort?
ORDER BY avg_fert_rate;



-- Final challenge (3)
-- determining the top 10 capital cities in Europe and the Americas in terms of a calculated percentage using city_proper_pop and metroarea_pop in cities
SELECT name, country_code, city_proper_pop, metroarea_pop,  
      city_proper_pop  / metroarea_pop * 100 AS city_perc
FROM cities
WHERE name IN
  (SELECT capital
   FROM countries
   WHERE (continent = 'Europe'
      OR continent LIKE '%America%'))
     AND metroarea_pop IS NOT NULL
ORDER BY city_perc DESC
LIMIT 10;