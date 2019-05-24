-- Chapter 1. Selecting columns

-- Onboarding | Tables ----
SELECT name FROM people;

-- Try running me!
SELECT 'DataCamp <3 SQL'
AS result;
  
  
-- SELECTing single columns ----
SELECT title FROM films;
SELECT release_year FROM films;
SELECT name FROM people;


-- SELECTing multiple columns ----
SELECT title FROM films;
SELECT title, release_year FROM films;
SELECT title, release_year, country FROM films;
SELECT * FROM films;


-- SELECT DISTINCT ----
SELECT DISTINCT country FROM films;
SELECT DISTINCT certification FROM films;
SELECT DISTINCT role FROM roles;


-- Learning to COUNT ----
-- You can test out queries here!
SELECT COUNT(*) FROM reviews;


-- Practice with COUNT
SELECT COUNT(*) FROM people;
SELECT COUNT(birthdate) FROM people;
SELECT COUNT(DISTINCT birthdate) FROM people;


-- Practice with COUNT
SELECT COUNT(*) FROM people;
SELECT COUNT(birthdate) FROM people;
SELECT COUNT(DISTINCT birthdate) FROM people;
SELECT COUNT(DISTINCT language) FROM films;
SELECT COUNT(DISTINCT country) FROM films;


-- Chapter 2. Filtering rows
-- Simple filtering of numeric values
SELECT * FROM films WHERE release_year = 2016;
SELECT COUNT(*) FROM films WHERE release_year < 2000;
SELECT title, release_year FROM films WHERE release_year > 2000;


-- Simple filtering of text
SELECT * FROM films WHERE language = 'French';
SELECT name, birthdate FROM people WHERE birthdate = '1974-11-11';
SELECT COUNT(*) FROM films WHERE language = 'Hindi';
SELECT * FROM films WHERE certification = 'R';


-- WHERE AND
SELECT title, release_year FROM films WHERE language = 'Spanish' AND release_year < 2000;
SELECT * FROM films WHERE language = 'Spanish' AND release_year > 2000;
SELECT * FROM films WHERE language = 'Spanish' AND release_year > 2000 AND release_year < 2010;


-- WHERE AND OR (2)
SELECT title, release_year FROM films WHERE (release_year > 1990 AND release_year <= 2000);
SELECT title, release_year FROM films WHERE (release_year >= 1990 AND release_year < 2000) AND (language = 'French' OR language = 'Spanish');
SELECT title, release_year FROM films WHERE (release_year >= 1990 AND release_year < 2000) AND (language = 'French' OR language = 'Spanish') AND (gross >2000000);


-- BETWEEN (2)
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000;
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000 AND budget > 100000000;
SELECT title, release_year from films WHERE release_year BETWEEN 1990 AND 2000 AND budget > 100000000 AND country = 'Spain';
SELECT title, release_year from films WHERE (language = 'Spanish' OR language = 'French') AND (release_year BETWEEN 1990 AND 2000) AND budget > 100000000;



-- WHERE IN
SELECT title, release_year from films WHERE (release_year IN (1990,2000)) AND duration > 120;
SELECT title, language FROM films WHERE language IN ('English', 'Spanish', 'French');
SELECT title, certification FROM films WHERE certification IN ('NC-17', 'R');


-- NULL and IS NULL
SELECT * from people WHERE 	deathdate is NULL;
SELECT title from films WHERE budget is NULL;
SELECT COUNT(*) from films WHERE language is NULL;


-- LIKE and NOT LIKE
-- The % wildcard will match zero, one, or many characters in text.
-- The _ wildcard will match a single character.

-- Get the names of all people whose names begin with 'B'. The pattern you need is 'B%'.
SELECT name FROM people WHERE name LIKE 'B%';
-- Get the names of people whose names have 'r' as the second letter. The pattern you need is '_r%'.
SELECT name FROM people WHERE name LIKE '_r%';
-- Get the names of people whose names don't start with A. The pattern you need is 'A%'.
SELECT name FROM people WHERE name NOT LIKE 'A%';



-- Chapter 3. Aggregate Functions
-- Aggregate functions
SELECT SUM(duration) from films;
SELECT AVG(duration) from films;
SELECT MIN(duration) from films;
SELECT MAX(duration) from films;


-- Aggregate functions practice
SELECT SUM(gross) from films;
SELECT AVG(gross) from films;
SELECT MIN(gross) from films;
SELECT MAX(gross) from films;


-- Combining aggregate functions with WHERE
SELECT SUM(gross) FROM films WHERE release_year >= 2000;
SELECT AVG(gross) FROM films WHERE title LIKE 'A%';
SELECT MIN(gross) FROM films WHERE release_year = 1994;
SELECT MAX(gross) FROM films WHERE release_year BETWEEN 2000 AND 2012;


-- A note on arithmetic
SELECT (10 / 3);
-- Answer is 3


-- It's AS simple AS aliasing
SELECT title, (gross - budget) AS net_profit FROM films;
SELECT title, (duration / 60.0) AS duration_hours FROM films;
SELECT AVG(duration / 60.0) AS avg_duration_hours FROM films;


-- Even more aliasing
-- Get the percentage of people who are no longer alive. Alias the result as percentage_dead. Remember to use 100.0 and not 100!
SELECT COUNT(deathdate) * 100.0 / COUNT(*) AS percentage_dead
FROM people;

-- Get the number of years between the oldest film and newest film. Alias the result as difference.
SELECT (MAX(release_year) - MIN(release_year)) AS difference 
FROM films;

-- Get the number of decades the films table covers. Alias the result as number_of_decades. The top half of your fraction should be enclosed in parentheses.
SELECT (MAX(release_year) - MIN(release_year)) / 10.0 
AS number_of_decades 
FROM films;



-- Chapter 4. Sorting, grouping and joins
-- Get the names of people from the people table, sorted alphabetically.
SELECT name 
FROM people
ORDER BY name;

-- Get the names of people, sorted by birth date.
SELECT name 
FROM people
ORDER BY birthdate;

-- Get the birth date and name for every person, in order of when they were born.
SELECT birthdate, name 
FROM people
ORDER BY birthdate;



-- Sorting single columns (2)
-- Get the title and release year of films released in 2000 or 2012, in the order they were released.
SELECT title, release_year
FROM films
WHERE release_year IN (2000,2012)
ORDER BY release_year;

-- Get all details for all films except those released in 2015 and order them by duration.
SELECT *
FROM films
WHERE release_year NOT IN (2015)
ORDER BY duration;

-- Get the title and gross box office earnings for movies which begin with the letter 'M' and order the results alphabetically.
SELECT title, gross
FROM films
WHERE title LIKE 'M%'
ORDER BY title;


-- Sorting single columns (DESC)
-- Get the IMDB score and film ID for every film, sorted from highest to lowest score.
SELECT 	imdb_score, film_id 
FROM reviews 
ORDER BY imdb_score DESC;

-- Get the title for every film, in reverse order.
SELECT title
FROM films
ORDER BY title DESC;

-- Get the title and duration for every film, in order of longest duration to shortest.
SELECT title, duration
FROM films
ORDER BY duration DESC;


-- Sorting multiple columns
-- Get the birth date and name of people in the people table, in order of when they were born and alphabetically by name.
SELECT birthdate, name
FROM people
ORDER BY birthdate, name;

-- Get the release year, duration, and title of films ordered by their release year and duration.
SELECT release_year, duration, title
FROM films
ORDER BY release_year, duration;

-- Get certifications, release years, and titles of films ordered by certification (alphabetically) and release year.
SELECT certification, release_year, title
FROM films
ORDER BY certification, release_year;

-- Get the names and birthdates of people ordered by birth date and name.
SELECT name, birthdate
FROM people
ORDER BY birthdate, name;



-- GROUP BY practice
-- Get the release year and count of films released in each year.
SELECT release_year, COUNT(*)
FROM films
GROUP BY release_year
ORDER BY  release_year;

-- Get the release year and average duration of all films, grouped by release year.
SELECT release_year, AVG(duration)
FROM films
GROUP BY release_year
ORDER BY  release_year;

-- Get the release year and largest budget for all films, grouped by release year.
SELECT release_year, MAX(budget)
FROM films
GROUP BY release_year
ORDER BY  release_year;

-- Get the IMDB score and count of film reviews for each IMDB rating in the reviews table.
SELECT imdb_score, COUNT(*)
FROM reviews
GROUP BY imdb_score;



-- GROUP BY practice (2)
-- Get the release year and lowest gross box office earnings per release year.
SELECT release_year, MIN(gross)
FROM films
GROUP BY release_year;

-- Get the language and total gross amount films in each language brought in at the box office.
SELECT language, SUM(gross)
FROM films
GROUP BY language;

-- Get the country and total budget spent making movies in each country.
SELECT country, SUM(budget)
FROM films
GROUP BY country;

-- Get the release year, country, and highest budget spent making a film for each year, for each country. Sort your results by release year and country.
SELECT release_year, country, MAX(budget)
FROM films
GROUP BY release_year, country
ORDER BY release_year, country;

-- Get the country, release year, and lowest amount grossed per release year per country. Order your results by country and release year.
SELECT country, release_year, MIN(gross)
FROM films
GROUP BY country, release_year
ORDER BY  country, release_year;


-- HAVING a great time
-- shows only those years in which more than 200 films were released.
SELECT release_year
FROM films
GROUP BY release_year
HAVING COUNT(title) > 200;

-- All together now
-- Get the release year, budget and box office earnings for each film in the films table.
SELECT release_year, budget, gross
FROM films;

-- Modify your query so that only results after 1990 are included.
SELECT release_year, budget, gross
FROM films
WHERE release_year > 1990;

-- Remove the budget and gross columns, and group your results by release year.
SELECT release_year
FROM films
WHERE release_year > 1990
GROUP BY release_year;


-- Modify your query to add in the average budget and average box office earnings for the results you have so far. Alias your results as avg_budget and avg_gross, respectively.
SELECT release_year, 
AVG(budget) AS avg_budget, 
AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year;

-- Modify your query so that only films with an average budget of greater than $60 million are included.
SELECT release_year, 
AVG(budget) AS avg_budget, 
AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000;


-- Finally, modify your query to order the results from highest average box office earnings to lowest.
SELECT release_year, 
AVG(budget) AS avg_budget, 
AVG(gross) AS avg_gross
FROM films
WHERE release_year > 1990
GROUP BY release_year
HAVING AVG(budget) > 60000000
ORDER BY avg_gross DESC;


-- All together now (2)
-- Get the name, average budget, and average box office take of countries that have made more than 10 films. Order the result by country name, and limit the number of results displayed to 5. You should alias the averages as avg_budget and avg_gross respectively.
SELECT
    country,
    AVG(budget) AS avg_budget, 
    AVG(gross)  AS avg_gross
FROM films
GROUP BY country
HAVING COUNT(title) > 10
ORDER BY country
LIMIT 5;


-- A taste of things to come
-- The query in the editor gets the IMDB score for the film To Kill a Mockingbird!
SELECT title, imdb_score
FROM films
JOIN reviews
ON films.id = reviews.id
WHERE title = 'To Kill a Mockingbird';