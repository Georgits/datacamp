use 07_data_driven_decision_making_in_sql;


-- CHAPTER 1: Introduction to business intelligence for a online movie rental database
-- Exploring the table renting
SELECT *  -- Select all
FROM renting;        -- From table renting

SELECT movie_id,  -- Select all columns needed to compute the average rating per movie
       rating
FROM renting;



-- Working with dates
SELECT *
FROM renting
WHERE date_renting = '2018-10-09'; -- Movies rented on October 9th, 2018

SELECT *
FROM renting
WHERE date_renting BETWEEN '2018-04-01' AND '2018-08-31'
ORDER BY date_renting; -- Order by recency in decreasing order




-- Selecting movies
SELECT *
FROM movies
WHERE genre <> 'Drama'; -- All genres except drama

SELECT *
FROM movies
WHERE title in ('Showtime', 'Love Actually', 'The Fighter'); -- Select all movies with the given titles

SELECT *
FROM movies
ORDER BY renting_price; -- Order the movies by increasing renting price


-- Select from renting
SELECT *
FROM renting
WHERE date_renting BETWEEN '2018-01-01' AND '2018-12-31' -- Renting in 2018
AND rating IS NOT NULL; -- Rating exists


-- Summarizing customer information
SELECT COUNT(*) -- Count the total number of customers
FROM customers
WHERE date_of_birth BETWEEN '1980-01-01' AND '1989-12-31'; -- Select customers born between 1980-01-01 and 1989-12-31

SELECT COUNT(*)   -- Count the total number of customers
FROM customers
WHERE country = 'Germany'; -- Select all customers from Germany

SELECT COUNT(DISTINCT country)   -- Count the number of countries
FROM customers;


-- Ratings of movie 25
SELECT MIN(CAST(rating AS DECIMAL)) min_rating, -- Calculate the minimum rating and use alias min_rating
	   MAX(CAST(rating AS DECIMAL)) max_rating, -- Calculate the maximum rating and use alias max_rating
	   AVG(CAST(rating AS DECIMAL)) avg_rating, -- Calculate the average rating and use alias avg_rating
	   COUNT(CAST(rating AS DECIMAL)) number_ratings -- Count the number of ratings and use alias number_ratings
FROM renting
WHERE movie_id = 25 AND rating <> ''; -- Select all records of the movie with ID 25


-- Examining annual rentals
SELECT * -- Select all records of movie rentals since January 1st 2019
FROM renting
WHERE date_renting >= '2019-01-01'; 

SELECT 
	COUNT(*), -- Count the total number of rented movies
	AVG(rating) -- Add the average rating
FROM renting
WHERE date_renting >= '2019-01-01';

SELECT 
	COUNT(*) number_renting, -- Give it the column name number_renting
	AVG(rating) average_rating  -- Give it the column name average_rating
FROM renting
WHERE date_renting >= '2019-01-01';

SELECT 
	COUNT(*) AS number_renting,
	AVG(rating) AS average_rating, 
    COUNT(rating) AS number_ratings -- Add the total number of ratings here.
FROM renting
WHERE date_renting >= '2019-01-01';





-- CHAPTER 2: Decision Making with simpel SQL queries

-- First account for each country.
SELECT country, -- For each country report the earliest date when an account was created
	MIN(date_account_start) AS first_account
FROM customers
GROUP BY country
ORDER BY first_account;


-- Average movie ratings
SELECT movie_id, 
       AVG(CAST(rating AS DECIMAL)) AS avg_rating,
       COUNT(CAST(rating AS DECIMAL)) AS number_ratings,
       COUNT(*) AS number_renting
FROM renting
GROUP BY movie_id
ORDER BY avg_rating DESC; -- Order by average rating in decreasing order


-- Average rating per customer
SELECT customer_id, -- Report the customer_id
      AVG(rating),  -- Report the average rating per customer
      COUNT(rating),  -- Report the number of ratings per customer
      COUNT(*)  -- Report the number of movie rentals per customer
FROM renting
GROUP BY customer_id
HAVING COUNT(*) > 7 -- Select only customers with more than 7 movie rentals
ORDER BY AVG(rating); -- Order by the average rating in ascending order


-- Join renting and customers
SELECT AVG(CAST(rating AS DECIMAL)) -- Average ratings of customers from Belgium
FROM renting AS r
LEFT JOIN customers AS c
ON r.customer_id = c.customer_id
WHERE c.country='Belgium';

-- Aggregating revenue, rentals and active customers
SELECT 
	SUM(m.renting_price), 
	COUNT(*), 
	COUNT(DISTINCT r.customer_id)
FROM renting AS r
LEFT JOIN movies AS m
ON r.movie_id = m.movie_id
-- Only look at movie rentals in 2018
WHERE date_renting BETWEEN '2018-01-01' AND '2018-12-31';



-- Movies and actors
SELECT a.name, -- Create a list of movie titles and actor names
       m.title
FROM actsin AS ai
LEFT JOIN movies AS m
ON m.movie_id = ai.movie_id
LEFT JOIN actors AS a
ON a.actor_id = ai.actor_id;


-- Income from movies
SELECT rm.title, -- Report the income from movie rentals for each movie 
       SUM(rm.renting_price) AS income_movie
FROM
       (SELECT m.title,  
               m.renting_price
       FROM renting AS r
       LEFT JOIN movies AS m
       ON r.movie_id=m.movie_id) AS rm
GROUP BY rm.title
ORDER BY SUM(rm.renting_price) DESC; -- Order the result by decreasing income



-- Age of actors from the USA
SELECT gender, -- Report for male and female actors from the USA 
       MAX(year_of_birth), -- The year of birth of the oldest actor
       MIN(year_of_birth) -- The year of birth of the youngest actor
FROM
   (SELECT * -- Use a subsequen SELECT to get all information about actors from the USA
    FROM actors
   WHERE nationality = 'USA') AS a -- Give the table the name a
GROUP BY gender;



-- Identify favorite movies for a group of customers
SELECT m.title, 
COUNT(*),
AVG(r.rating)
FROM renting AS r
LEFT JOIN customers AS c
ON c.customer_id = r.customer_id
LEFT JOIN movies AS m
ON m.movie_id = r.movie_id
WHERE c.date_of_birth BETWEEN '1970-01-01' AND '1979-12-31'
GROUP BY m.title
HAVING COUNT(*) <> 1 -- Remove movies with only one rental
ORDER BY AVG(r.rating); -- Order with highest rating first



-- Identify favorite actors for Spain
SELECT a.name,  c.gender,
       COUNT(*) AS number_views, 
       AVG(r.rating) AS avg_rating
FROM renting as r
LEFT JOIN customers AS c
ON r.customer_id = c.customer_id
LEFT JOIN actsin as ai
ON r.movie_id = ai.movie_id
LEFT JOIN actors as a
ON ai.actor_id = a.actor_id
WHERE c.country = 'Spain' -- Select only customers from Spain
GROUP BY a.name, c.gender
HAVING AVG(r.rating) IS NOT NULL 
  AND COUNT(*) > 5 
ORDER BY avg_rating DESC, number_views DESC;


-- KPIs per country
SELECT 
	country,                    -- For each country report
	COUNT(*) AS number_renting, -- The number of movie rentals
	AVG(rating) AS average_rating, -- The average rating
	SUM(renting_price) AS revenue         -- The revenue from movie rentals
FROM renting AS r
LEFT JOIN customers AS c
ON c.customer_id = r.customer_id
LEFT JOIN movies AS m
ON m.movie_id = r.movie_id
WHERE date_renting >= '2019-01-01'
GROUP BY country;