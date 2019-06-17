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


SELECT MIN(CAST(rating AS DECIMAL)) FROM renting
WHERE movie_id = 25 AND rating <> '';