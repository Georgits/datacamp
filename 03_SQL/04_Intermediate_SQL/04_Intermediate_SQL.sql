use 04_intermediate_sql;

SELECT
	-- Select the team long name and team API id
	team_long_name,
	team_api_id
FROM team
-- Only include FC Schalke 04 and FC Bayern Munich
WHERE team_long_name IN ('FC Schalke 04', 'FC Bayern Munich');


-- Identify the home team as Bayern Munich, Schalke 04, or neither
SELECT 
	CASE WHEN home_team_api_id = 10189 THEN 'FC Schalke 04'
        WHEN home_team_api_id = 9823 THEN 'FC Bayern Munich'
         ELSE 'Other' END AS home_team,
	COUNT(id) AS total_matches
FROM matches
-- Group by the CASE statement alias
GROUP BY home_team;

-- CASE statements comparing column values
SELECT 
	-- Select the date of the match
	date,
	-- Identify home wins, losses, or ties
	CASE WHEN home_team_goal > away_team_goal THEN 'Home win!'
        WHEN home_team_goal < away_team_goal THEN 'Home loss :(' 
        ELSE 'Tie' END AS outcome
FROM matches;


SELECT 
	m.date,
	-- Select the team long name column and call it 'opponent'
	t.team_long_name AS opponent, 
	-- Complete the CASE statement with an alias
	CASE WHEN m.home_team_goal > m.away_team_goal THEN 'Home win!'
        WHEN m.home_team_goal < m.away_team_goal THEN 'Home loss :('
        ELSE 'Tie' END AS outcome
FROM matches AS m
-- Left join teams_spain onto matches_spain
LEFT JOIN team AS t
ON m.away_team_api_id = t.team_api_id;


SELECT 
	m.date,
	t.team_long_name AS opponent,
    -- Complete the CASE statement with an alias
	CASE WHEN m.home_team_goal > m.away_team_goal THEN 'Home win!'
        WHEN m.home_team_goal < m.away_team_goal THEN 'Home loss :('
        ELSE 'Tie' END AS outcome 
FROM matches AS m
-- Left join teams_spain onto matches_spain
LEFT JOIN team AS t
ON m.away_team_api_id = t.team_api_id
-- Filter for Barcelona as the home team
WHERE m.home_team_api_id = '8634';



-- CASE statements comparing two column values part 2
-- Select matches where Barcelona was the away team
SELECT 
	m.date,
	t.team_long_name AS opponent,
    -- Complete the CASE statement with an alias
	CASE WHEN m.home_team_goal > m.away_team_goal THEN 'Home win!'
        WHEN m.home_team_goal < m.away_team_goal THEN 'Home loss :('
        ELSE 'Tie' END AS outcome 
FROM matches AS m
-- Left join teams_spain onto matches_spain
LEFT JOIN team AS t
ON m.away_team_api_id = t.team_api_id
-- Filter for Barcelona as the home team
WHERE m.away_team_api_id = 8634;




-- In CASE of rivalry
SELECT 
	date,
	-- Identify the home team as Barcelona or Real Madrid
	CASE WHEN home_team_api_id = 8634 THEN 'FC Barcelona' 
        ELSE 'Real Madrid CF' END AS home,
    -- Identify the away team as Barcelona or Real Madrid
	CASE WHEN away_team_api_id = 8634 THEN 'FC Barcelona' 
        ELSE 'Real Madrid CF' END AS away
FROM matches
WHERE (away_team_api_id = 8634 OR home_team_api_id = 8634)
      AND (away_team_api_id = 8633 OR home_team_api_id = 8633);
      
      
      
      
SELECT 
	date,
	CASE WHEN home_team_api_id = 8634 THEN 'FC Barcelona' 
         ELSE 'Real Madrid CF' END as home,
	CASE WHEN away_team_api_id = 8634 THEN 'FC Barcelona' 
         ELSE 'Real Madrid CF' END as away,
	-- Identify all possible match outcomes
	CASE WHEN home_team_goal > away_team_goal AND home_team_api_id = 8634 THEN 'Barcelona win!'
        WHEN home_team_goal > away_team_goal AND home_team_api_id = 8633 THEN 'Real Madrid win!'
        WHEN home_team_goal < away_team_goal AND away_team_api_id = 8634 THEN 'Barcelona win!'
        WHEN home_team_goal < away_team_goal AND away_team_api_id = 8633 THEN 'Real Madrid win!'
        ELSE 'Tie!' END AS outcome
FROM matches
WHERE (away_team_api_id = 8634 OR home_team_api_id = 8634)
      AND (away_team_api_id = 8633 OR home_team_api_id = 8633);
      
      
      
-- Filtering your CASE statement
-- Select team_long_name and team_api_id from team
SELECT
	team_long_name,
	team_api_id
FROM team
-- Filter for team name
WHERE team_long_name = 'Bologna';


-- Select the season and date columns
SELECT 
	season,
	date,
    -- Identify when Bologna won a match
	CASE WHEN home_team_api_id = 9857 AND  home_team_goal > away_team_goal THEN 'Bologna Win'
		WHEN away_team_api_id = 9857 AND away_team_goal > home_team_goal THEN 'Bologna Win' 
		END AS outcome
FROM matches;



-- Select the season, date, home_goal, and away_goal columns
SELECT 
	season,
    date,
	home_team_goal,
	away_team_goal
FROM matches
WHERE 
-- Exclude games not won by Bologna
	CASE WHEN home_team_api_id = 9857 AND  home_team_goal> away_team_goal THEN 'Bologna Win'
		WHEN away_team_api_id = 9857 AND away_team_goal >home_team_goal THEN 'Bologna Win' 
		END IS NOT NULL;
        
        
-- COUNT using CASE WHEN
SELECT 	c.name AS country,
    -- Count games from the 2012/2013 season
	COUNT(CASE WHEN m.season = '2012/2013' 
        	THEN m.id ELSE NULL END) AS matches_2012_2013
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY country;


SELECT 	c.name AS country,
    -- Count matches in each of the 3 seasons
	COUNT(CASE WHEN m.season = '2012/2013' THEN m.id END) AS matches_2012_2013,
	COUNT(CASE WHEN m.season = '2013/2014' THEN m.id END) AS matches_2013_2014,
	COUNT(CASE WHEN m.season = '2014/2015' THEN m.id END) AS matches_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY country;


-- COUNT and CASE WHEN with multiple conditions
SELECT 
	c.name AS country,
    -- Sum the total records in each season where the home team won
	SUM(CASE WHEN m.season = '2012/2013' AND m.home_team_goal > m.away_team_goal 
        THEN 1 ELSE 0 END) AS matches_2012_2013,
 	SUM(CASE WHEN m.season = '2013/2014' AND m.home_team_goal > m.away_team_goal 
        THEN 1 ELSE 0 END) AS matches_2013_2014,
	SUM(CASE WHEN m.season = '2014/2015' AND m.home_team_goal > m.away_team_goal 
        THEN 1 ELSE 0 END) AS matches_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
-- Group by country name alias
GROUP BY country;



-- Calculating percent with CASE and AVG
SELECT 
    c.name AS country,
    -- Count the home wins, away wins, and ties in each country
	COUNT(CASE WHEN m.home_team_goal > m.away_team_goal THEN m.id 
        END) AS home_wins,
	COUNT(CASE WHEN m.home_team_goal < m.away_team_goal THEN m.id 
        END) AS away_wins,
	COUNT(CASE WHEN m.home_team_goal = m.away_team_goal THEN m.id 
        END) AS ties
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country;


SELECT 
	c.name AS country,
    -- Calculate the percentage of tied games in each season
	AVG(CASE WHEN m.season='2013/2014' AND m.home_team_goal = m.away_team_goal THEN 1
			WHEN m.season='2013/2014' AND m.home_team_goal != m.away_team_goal THEN 0
			END) AS ties_2013_2014,
	AVG(CASE WHEN m.season='2014/2015' AND m.home_team_goal = m.away_team_goal THEN 1
			WHEN m.season='2014/2015' AND m.home_team_goal != m.away_team_goal THEN 0
			END) AS ties_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country;



SELECT 
	c.name AS country,
    -- Round the percentage of tied games to 2 decimal points
	ROUND(AVG(CASE WHEN m.season='2013/2014' AND m.home_team_goal = m.away_team_goal THEN 1
			 WHEN m.season='2013/2014' AND m.home_team_goal != m.away_team_goal THEN 0
			 END),2) AS pct_ties_2013_2014,
	ROUND(AVG(CASE WHEN m.season='2014/2015' AND m.home_team_goal = m.away_team_goal THEN 1
			 WHEN m.season='2014/2015' AND m.home_team_goal != m.away_team_goal THEN 0
			 END),2) AS pct_ties_2014_2015
FROM country AS c
LEFT JOIN matches AS m
ON c.id = m.country_id
GROUP BY country;





-- CHAPTER 2: Short and Simple Subqueries
-- Filtering using scalar subqueries
-- Select the average of home + away goals, multiplied by 3
SELECT 
	3 * AVG(home_team_goal + away_team_goal)
FROM matches;


SELECT 
	-- Select the date, home goals, and away goals scored
    date,
	home_team_goal,
	away_team_goal
FROM  matches
-- Filter for matches where total goals exceeds 3x the average
WHERE (home_team_goal + away_team_goal) > 
       (SELECT 3 * AVG(home_team_goal + away_team_goal)
        FROM matches); 


-- Filtering using a subquery with a list
SELECT 
	-- Select the team long and short names
	team_long_name,
	team_short_name
FROM team 
-- Exclude all values from the subquery
WHERE team_api_id NOT IN
     (SELECT DISTINCT home_team_api_ID  FROM matches);


-- Filtering with more complex subquery conditions
SELECT
	-- Select the team long and short names
	team_long_name,
	team_short_name
FROM team
-- Filter for teams with 8 or more home goals
WHERE team_api_id IN
	  (SELECT home_team_api_ID 
       FROM matches
       WHERE home_team_goal >= 8);


-- Joining Subqueries in FROM
SELECT 
	-- Select the country ID and match ID
	country_id, 
    id 
FROM matches
-- Filter for matches with 10 or more goals in total
WHERE (home_team_goal + away_team_goal) >= 10;


SELECT
	-- Select country name and the count match IDs
    c.name AS country_name,
    COUNT(sub.id) AS matches
FROM country AS c
-- Inner join the subquery onto country
-- Select the country id and match id columns
INNER JOIN (SELECT country_id, id 
            FROM matches
            -- Filter the subquery by matches with 10+ goals
            WHERE (home_team_goal + away_team_goal) >= 10) AS sub
ON c.id = sub.country_id
GROUP BY country_name;


-- Building on Subqueries in FROM
SELECT
	-- Select country, date, home, and away goals from the subquery
    country,
    date,
    home_team_goal,
    away_team_goal
FROM 
	-- Select country name, date, and total goals in the subquery
	(SELECT c.name AS country, 
     	    m.date, 
     		m.home_team_goal, 
     		m.away_team_goal,
           (m.home_team_goal + m.away_team_goal) AS total_goals
    FROM matches AS m
    LEFT JOIN country AS c
    ON m.country_id = c.id) AS subquery
-- Filter by total goals scored in the main query
WHERE total_goals >= 10;


-- Add a subquery to the SELECT clause
SELECT 
	l.name AS league,
    -- Select and round the league's total goals
    ROUND(AVG(m.home_team_goal + m.away_team_goal), 2) AS avg_goals,
    -- Select & round the average total goals for the season
    (SELECT ROUND(AVG(home_team_goal + away_team_goal), 2) 
     FROM matches
     WHERE season = '2013/2014') AS overall_avg
FROM league AS l
LEFT JOIN matches AS m
ON l.country_id = m.country_id
-- Filter for the 2013/2014 season
WHERE season = '2013/2014'
GROUP BY l.name;


-- Subqueries in Select for Calculations
SELECT
	-- Select the league name and average goals scored
	l.name AS league,
	ROUND(AVG(m.home_team_goal + m.away_team_goal),2) AS avg_goals,
    -- Subtract the overall average from the league average
	ROUND(AVG(m.home_team_goal + m.away_team_goal) - 
		(SELECT AVG(home_team_goal + away_team_goal)
		 FROM matches 
         WHERE season = '2013/2014'),2) AS diff
FROM league AS l
LEFT JOIN matches AS m
ON l.country_id = m.country_id
-- Only include 2013/2014 results
WHERE season = '2013/2014'
GROUP BY l.name;


-- ALL the Subqueries EVERYWHERE
SELECT 
	-- Select the stage and average goals for each stage
	m.stage,
    ROUND(AVG(m.home_team_goal + m.away_team_goal),2) AS avg_goals,
    -- Select the average overall goals for the 2012/2013 season
    ROUND((SELECT AVG(home_team_goal + away_team_goal) 
           FROM matches 
           WHERE season = '2012/2013'),2) AS overall
FROM matches AS m
-- Filter for the 2012/2013 season
WHERE season = '2012/2013'
-- Group by stage
GROUP BY m.stage;



-- Add a subquery in FROM
SELECT 
	-- Select the stage and average goals from the subquery
	s.stage,
	ROUND(s.avg_goals,2) AS avg_goals
FROM 
	-- Select the stage and average goals in 2012/2013
	(SELECT
		 stage,
         AVG(home_team_goal + away_team_goal) AS avg_goals
	 FROM matches
	 WHERE season = '2012/2013'
	 GROUP BY stage) AS s
WHERE 
	-- Filter the main query using the subquery
	s.avg_goals > (SELECT AVG(home_team_goal + away_team_goal) 
                    FROM matches WHERE season = '2012/2013');



-- Add a subquery in SELECT
SELECT 
	-- Select the stage and average goals from s
	s.stage,
    ROUND(s.avg_goals,2) AS avg_goal,
    -- Select the overall average for 2012/2013
    (SELECT AVG(home_team_goal + away_team_goal) FROM matches WHERE season = '2012/2013') AS overall_avg
FROM 
	-- Select the stage and average goals in 2012/2013 from match
	(SELECT country
		 stage,
         AVG(home_team_goal + away_team_goal) AS avg_goals
	 FROM matches
	 WHERE season = '2012/2013'
	 GROUP BY stage) AS s
WHERE 
	-- Filter the main query using the subquery
	s.avg_goals > (SELECT AVG(home_team_goal + away_team_goal) 
                    FROM matches WHERE season = '2012/2013');



-- CHAPTER 3: Correlated Queries, Nested Queries, and Common Table Expressions

-- Basic Correlated Subqueries
SELECT 
	-- Select country ID, date, home, and away goals from match
	main.country_id,
    date,
    main.home_team_goal, 
    away_team_goal
FROM matches AS main
WHERE 
	-- Filter the main query by the subquery
	(home_team_goal + away_team_goal) > 
        (SELECT AVG((sub.home_team_goal + sub.away_team_goal) * 3)
         FROM matches AS sub
         -- Join the main query to the subquery in WHERE
         WHERE main.country_id = sub.country_id);


-- Correlated subquery with multiple conditions
SELECT 
	-- Select country ID, date, home, and away goals from match
	main.country_id,
    main.date,
    main.home_team_goal,
    main.away_team_goal
FROM matches AS main
WHERE 
	-- Filter for matches with the highest number of goals scored
	(home_team_goal + away_team_goal) = 
        (SELECT MAX(sub.home_team_goal + sub.away_team_goal)
         FROM matches AS sub
         WHERE main.country_id = sub.country_id
               AND main.season = sub.season);
               
               
               
-- Nested simple subqueries
SELECT
	-- Select the season and max goals scored in a match
	season,
    MAX(home_team_goal + away_team_goal) AS max_goals,
    -- Select the overall max goals scored in a match
   (SELECT MAX(home_team_goal + away_team_goal) FROM matches) AS overall_max_goals,
   -- Select the max number of goals scored in any match in July
   (SELECT MAX(home_team_goal + away_team_goal) 
    FROM matches
    WHERE id IN (
          SELECT id FROM matches WHERE EXTRACT(MONTH FROM date) = 07)) AS july_max_goals
FROM matches
GROUP BY season;


-- Nest a subquery in FROM
-- Select matches where a team scored 5+ goals
SELECT
	country_id,
    season,
	id
FROM matches
WHERE home_team_goal >=5 OR away_team_goal >=5;


-- Count match ids
SELECT
    country_id,
    season,
    COUNT(id) AS matches
-- Set up and alias the subquery
FROM (
	SELECT
    	country_id,
    	season,
    	id
	FROM matches
	WHERE home_team_goal >= 5 OR away_team_goal >= 5) AS subquery
-- Group by country_id and season
GROUP BY country_id, season;


SELECT
	c.name AS country,
    -- Calculate the average matches per season
	AVG(outer_s.matches) AS avg_seasonal_high_scores
FROM country AS c
-- Left join outer_s to country
LEFT OUTER JOIN (
  SELECT country_id, season,
         COUNT(id) AS matches
  FROM (
    SELECT country_id, season, id
	FROM matches
	WHERE home_team_goal >= 5 OR away_team_goal >= 5) AS inner_s
  -- Close parentheses and alias the subquery
  GROUP BY country_id, season ) AS outer_s
ON c.id = outer_s.country_id
GROUP BY country;



-- Clean up with CTEs
-- Set up your CTE
WITH match_list AS (
    SELECT 
  		country_id, 
  		id
    FROM matches
    WHERE (home_team_goal + away_team_goal) >= 10)
-- Select league and count of matches from the CTE
SELECT
    l.name AS league,
    COUNT(match_list.id) AS matches
FROM league AS l
-- Join the CTE to the league table
LEFT JOIN match_list ON l.id = match_list.country_id
GROUP BY l.name;



-- Organizing with CTEs
-- Set up your CTE
WITH match_list AS (
  -- Select the league, date, home, and away goals
    SELECT 
  		l.name AS league, 
     	m.date, 
  		m.home_team_goal, 
  		m.away_team_goal,
       (m.home_team_goal + m.away_team_goal) AS total_goals
    FROM matches AS m
    LEFT JOIN league as l ON m.country_id = l.id)
-- Select the league, date, home, and away goals from the CTE
SELECT league, date, home_team_goal, away_team_goal
FROM match_list
-- Filter by total goals
WHERE total_goals >=5;


-- CTEs with nested subqueries
-- Set up your CTE
WITH match_list AS (
    SELECT 
  		country_id,
  	   (home_team_goal + away_team_goal) AS goals
    FROM matches
  	-- Create a list of match IDs to filter data in the CTE
    WHERE id IN (
       SELECT id
       FROM matches
       WHERE season = '2008/2009' AND EXTRACT(MONTH FROM date) = 08))
-- Select the league name and average of goals in the CTE
SELECT 
	l.name,
    AVG(match_list.goals)
FROM league AS l
-- Join the CTE onto the league table
LEFT JOIN match_list ON l.id = match_list.country_id
GROUP BY l.name;



-- Get team names with a subquery
SELECT 
	m.id, 
    t.team_long_name AS hometeam
-- Left join team to match
FROM matches AS m
LEFT JOIN team as t
ON m.home_team_api_id = t.team_api_id;


SELECT
	m.date,
    -- Get the home and away team names
    hometeam,
    awayteam,
    m.home_team_goal,
    m.away_team_goal
FROM matches AS m
-- Join the home subquery to the match table
LEFT JOIN (
  SELECT matches.id, team.team_long_name AS hometeam
  FROM matches
  LEFT JOIN team
  ON matches.home_team_api_id = team.team_api_id) AS home
ON home.id = m.id
-- Join the away subquery to the match table
LEFT JOIN (
  SELECT matches.id, team.team_long_name AS awayteam
  FROM matches
  LEFT JOIN team
  -- Get the away team ID in the subquery
  ON matches.away_team_api_id = team.team_api_id) AS away
ON away.id = m.id;



-- Get team names with correlated subqueries
SELECT
    m.date,
   (SELECT team_long_name
    FROM team AS t
    -- Connect the team to the match table
    WHERE t.team_api_id = m.home_team_api_id) AS hometeam
FROM matches AS m;


SELECT
    m.date,
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.home_team_api_id) AS hometeam,
    -- Connect the team to the match table
    (SELECT team_long_name
     FROM team AS t
     WHERE t.team_api_id = m.away_team_api_id) AS awayteam,
    -- Select home and away goals
     m.home_team_goal,
     m.away_team_goal
FROM matches AS m;


-- Get team names with CTEs
SELECT 
	-- Select match id and team long name
    m.id, 
    t.team_long_name AS hometeam
FROM matches AS m
-- Join team to match using team_api_id and hometeam_id
LEFT JOIN team AS t 
ON m.home_team_api_id = t.team_api_id;


-- Declare the home CTE
WITH home AS (
	SELECT m.id, t.team_long_name AS hometeam
	FROM matches AS m
	LEFT JOIN team AS t 
	ON m.home_team_api_id = t.team_api_id)
-- Select everything from home
SELECT *
FROM home;


WITH home AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS hometeam, m.home_team_goal
  FROM matches AS m
  LEFT JOIN team AS t 
  ON m.home_team_api_id = t.team_api_id),
-- Declare and set up the away CTE
away AS (
  SELECT m.id, m.date, 
  		 t.team_long_name AS awayteam, m.away_team_goal
  FROM matches AS m
  LEFT JOIN team AS t 
  ON m.away_team_api_id = t.team_api_id)
-- Select date, home_goal, and away_goal
SELECT 
	home.date,
    home.hometeam,
    away.awayteam,
    home.home_team_goal,
    away.away_team_goal
-- Join away and home on the id column
FROM home
INNER JOIN away
ON home.id = away.id;