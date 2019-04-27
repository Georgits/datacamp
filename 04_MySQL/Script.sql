
/* cd C:\Program Files\MySQL\MySQL Server 8.0\bin */

mysql -u root -p:
quit;

create database bank;


/* http://www.aodba.com/create-user-grant-permissions-mysql/ */

CREATE USER 'lrng'@'localhost' IDENTIFIED BY 'myfirst';
GRANT ALL PRIVILEGES ON bank . * TO 'lrng'@'localhost';
quit;

msql -u lrngsql -p;
use bank;

source C:\Users\D91067\Desktop\MySQL\LearningSQLExample.sql;

mysql -u lrngsql -p bank

SELECT now();
SELECT now() FROM dual;

SHOW CHARACTER SET;


CREATE database foreign_sales character set utf8;


/* TABLE CREATION */


CREATE TABLE person
	(person_id SMALLINT UNSIGNED,
	fname VARCHAR(20),
	lname VARCHAR(20),
	gender CHAR1(1),
	birth_date DATE,
	street VARCHAR(30),
	city VARCHAR(20),
	state VARCHAR(20),
	country VARCHAR(20),
	postal_code VARCHAR(20),
	CONSTRAINT pk_person PRIMARY KEY (person_id)
	);
	
	
/* Adding contraint on a column gender */	
CREATE TABLE person
	(person_id SMALLINT UNSIGNED,
	fname VARCHAR(20),
	lname VARCHAR(20),
	gender ENUM('M','F'),
	birth_date DATE,
	street VARCHAR(30),
	city VARCHAR(20),
	state VARCHAR(20),
	country VARCHAR(20),
	postal_code VARCHAR(20),
	CONSTRAINT pk_person PRIMARY KEY (person_id)
	);
	
	
DESC person;


ALTER TABLE person MODIFY person_id SMALLINT UNSIGNED AUTO_INCREMENT;
DESC person;


CREATE TABLE favorite_food 
	(person_id SMALLINT UNSIGNED,
	food VARCHAR(20),
	CONSTRAINT pk_favorite_food PRIMARY KEY (person_id, food),
	CONSTRAINT fk_fav_food_person_id FOREIGN KEY (person_id)
		REFERENCES person (person_id)
	);

	
DESC favorite_food;




/* POPULATING AND MODIFYING TABLES */
INSERT INTO person
	(person_id, fname, lname, gender, birth_date)
	VALUES(null, 'William', 'Turner', 'M', '1972-05-27');
	

SELECT person_id, fname, lname, birth_date
		FROM person;
		
		
SELECT person_id, fname, lname, birth_date
		FROM person
		WHERE person_id = 1;
		
SELECT person_id, fname, lname, birth_date
		FROM person
		WHERE lname = 'Turner';

		
INSERT INTO favorite_food (person_id, food)
	VALUES(1, 'pizza');
	
INSERT INTO favorite_food (person_id, food)
	VALUES(1, 'cookies');
	
INSERT INTO favorite_food (person_id, food)
	VALUES(1, 'nachos');
	
SELECT food
	FROM favorite_food
	WHERE person_id = 1
	ORDER BY food;
	
	
	
INSERT INTO person
	(person_id, fname, lname, gender, birth_date, street, city, state, country, postal_code)
	VALUES(null, 'Susan', 'Smith', 'F', '1975-11-02', '23 Maple St.', 'Arlington', 'VA', 'USA', '20220');
	
SELECT person_id, fname, lname, birth_date
	FROM person;

SELECT * FROM favorite_food
	FOR XML AUTO;
	

UPDATE person
	SET street = '1225 Tremont St.',
	city = 'Boston',
	state='MA',
	country='USA',
	postal_code='02138'
	WHERE person_id=1;

	
DELETE FROM person
	WHERE person_id=2;
	
	
/*NONUNIQUE PRIMARY KEY */
INSERT INTO person
	(person_id, fname, lname, gender, birth_date)
	VALUES(1, 'Charles', 'Fulton', 'M', '1968-01-15');
	
/*NONEXISTENT FOREIGN KEY */
INSERT INTO favorite_food
	(person_id, food)
	VALUES(999, 'lasagna');

/*COLUMN VALUE VIOLATIONS*/
UPDATE person
	SET gender = 'Z'
	WHERE person_id=1;
	
/*INVALID DATE CONVERSIONS*/
UPDATE person
	SET birth_date = 'DEC-21-1980'
	WHERE person_id=1;

/*VALID DATE CONVERSIONS*/
UPDATE person
	SET birth_date = str_to_date('DEC-21-1980', '%b-%d-%Y')
	WHERE person_id=1;
	

SHOW TABLES;
DESC bank.account;





DROP TABLE person;
DROP TABLE favorite_food;


select fname, lname	FROM employee;

use bank;
select fname, lname	FROM employee;


select * FROM bank.department;
select name FROM department;
SELECT emp_id, 
	'ACTIVE', 
    emp_id * 3.14159, 
    UPPER(lname) 
    FROM employee;
    
    
SELECT VERSION(), USER(), DATABASE();


SELECT emp_id, 
	'ACTIVE' AS status, 
    emp_id * 3.14159 AS empid_x_pi, 
    UPPER(lname) AS last_name_upper 
    FROM employee;
    
    
    
/* Removing Duplicates */
SELECT cust_id FROM account;
SELECT distinct cust_id FROM account;


/* SUBQUERIES */
SELECT e.emp_id, e.fname, e.lname
	FROM (SELECT emp_id, fname, lname, start_date, title FROM employee) e;
    
    
/* VIEWS */
CREATE VIEW employee_vw AS
	SELECT emp_id, fname, lname, YEAR(start_date) AS start_year
    FROM employee;

SELECT emp_id, start_year FROM  employee_vw;

SELECT employee.emp_id, employee.fname, employee.lname, department.name AS dept_name
	FROM employee INNER JOIN department
    ON employee.dept_id = department.dept_id;

SELECT e.emp_id, e.fname, e.lname, d.name AS dept_name
	FROM employee AS e INNER JOIN department AS d 
    ON e.dept_id = d.dept_id;


SELECT emp_id, fname, lname, start_date, title 
	FROM employee
    WHERE title = 'Head Teller'
		AND start_date >'2001-01-01';
        
SELECT emp_id, fname, lname, start_date, title 
	FROM employee
    WHERE (title = 'Head Teller' AND start_date >'2001-01-01')
		OR (title = 'Teller' AND start_date >'2002-01-01');
        
        
SELECT d.name, count(e.emp_id) AS num_employees
	FROM department AS d INNER JOIN employee AS e
		ON d.dept_id = e.dept_id
    GROUP BY d.name
    HAVING count(num_employees) > 2;
    
SELECT open_emp_id, product_cd
	FROM account
    ORDER BY open_emp_id, product_cd;
    
    
SELECT account_id, product_cd, open_date, avail_balance
	FROM account
    ORDER BY avail_balance DESC;
    
SELECT cust_id, cust_type_cd, city, state, fed_id
	FROM customer
    ORDER BY RIGHT(fed_id, 3);
    
SELECT emp_id, title, start_date, fname, lname
	FROM employee
    ORDER BY 2, 5;
    
/* exercise 3-1 */
SELECT emp_id, fname, lname
	FROM employee
    ORDER BY lname, fname;
    
/* exercise 3-2 */
desc account;
SELECT account_id, cust_id, avail_balance
	FROM account
    WHERE status = 'ACTIVE' AND avail_balance > 2500;

/* exercise 3-3 */
desc account;
SELECT distinct open_emp_id
	FROM account;
    


/* exercise 3-4 */
desc product;
SELECT * FROM product;
SELECT p.product_cd, a.cust_id, a.avail_balance
	FROM product AS p INNER JOIN account AS a
		ON p.product_cd = a.product_cd
	WHERE p.product_type_cd = 'ACCOUNT'
    ORDER BY p.product_cd, a.cust_id;






/* CHAPTER 4: FILTERING */
SELECT pt.name AS product_type, p.name AS product
	FROM product AS p INNER JOIN product_type AS pt
		ON p.product_type_cd = pt.product_type_cd
	WHERE pt.name = 'Customer Accounts';
    
SELECT pt.name AS product_type, p.name AS product
	FROM product AS p INNER JOIN product_type AS pt
		ON p.product_type_cd = pt.product_type_cd
	WHERE pt.name <> 'Customer Accounts';
    
SELECT emp_id, fname, lname, start_date
	FROM employee
    WHERE start_date BETWEEN '2001-01-01' AND '2005-01-01';
    
SELECT cust_id, fed_id
	FROM customer
    WHERE cust_type_cd = 'I'
		AND fed_id BETWEEN '500-00-0000' AND '999-99-9999';
        
SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE product_cd IN ('CHK', 'SAV', 'CD', 'MM');
    
SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE product_cd NOT IN ('CHK', 'SAV', 'CD', 'MM');

SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE product_cd IN (SELECT product_cd FROM product
							WHERE product_type_cd = 'ACCOUNT');
                           
SELECT emp_id, fname, lname
	FROM employee
    WHERE LEFT(lname, 1) = 'T';
    
SELECT lname FROM employee WHERE lname LIKE '_a%e%';

SELECT emp_id, fname, lname 
	FROM employee 
    WHERE lname LIKE 'F%' OR lname LIKE 'G%';
    
SELECT emp_id, fname, lname 
	FROM employee 
    WHERE lname REGEXP '^[FG]';
    
SELECT emp_id, fname, lname, superior_emp_id
	FROM employee 
    WHERE superior_emp_id IS NULL;
    
SELECT emp_id, fname, lname, superior_emp_id
	FROM employee 
    WHERE superior_emp_id !=6 OR superior_emp_id IS NULL;
    
    
/* EXERCISE 4-3 */
desc account;
SELECT * FROM account WHERE YEAR(open_date) = 2002;

SELECT account_id, open_date
	FROM account
    WHERE open_date BETWEEN '2002-01-01' AND '2002-12-31';

/* EXERCISE 4-4 */
desc individual;
SELECT cust_id, lname, fname 
 FROM individual
 WHERE lname LIKE '_a%e%';







/* CHAPTER 5: Querying Multiple Tables */
DESC employee;
DESC department; 


/* INNER JOINS */
SELECT e.fname, e.lname, d.name
	FROM employee AS e INNER JOIN department AS d
		ON e.dept_id = d.dept_id;
        
SELECT e.fname, e.lname, d.name
	FROM employee AS e INNER JOIN department AS d
		USING (dept_id);
        
SELECT e.fname, e.lname, d.name
	FROM employee AS e INNER JOIN department AS d
	WHERE e.dept_id = d.dept_id;

SELECT a.account_id, a.cust_id, a.open_date, a.product_cd
	FROM account AS a INNER JOIN employee AS e
		ON a.open_emp_id = e.emp_id
        INNER JOIN branch AS b
        ON e.assigned_branch_id=b.branch_id
	WHERE e.start_date < '2007-01-01'
		AND (e.title = 'Teller' OR e.title = 'Head Teller')
        AND b.name = 'Woburn Branch';


/* JOINING THREE OR MORE TABLES */
SELECT a.account_id, c.fed_id, e.fname, e.lname
	FROM account AS a INNER JOIN customer AS c
		ON a.cust_id = c.cust_id
        INNER JOIN employee AS e
        ON a.open_emp_id = e.emp_id
	WHERE c.cust_type_cd = 'B';
    
    
    
SELECT a.account_id, a.cust_id, a.open_date, a.product_cd
	FROM account AS a INNER JOIN
		(SELECT emp_id, assigned_branch_id
			FROM employee
            WHERE start_date < '2007-01-01'
				AND (title = 'Teller' OR title = 'Head Teller')) AS e
		ON a.open_emp_id = e.emp_id
        INNER JOIN
			(SELECT branch_id
				FROM branch
                WHERE name = 'Woburn Branch') AS b
		ON e.assigned_branch_id = b.branch_id;
        
/* USING THE SAME TABLE TWICE */
SELECT a.account_id, e.emp_id,
	b_a.name AS open_branch,
    b_e.name AS emp_branch
    FROM account AS a INNER JOIN branch AS b_a
		ON a.open_branch_id = b_a.branch_id
        INNER JOIN employee AS e
        ON a.open_emp_id = e.emp_id
        INNER JOIN branch AS b_e
        ON e.assigned_branch_id = b_e.branch_id
	WHERE a.product_cd = 'CHK';
    
    
/* SELF-JOINS */
SELECT e.fname, e.lname,
		e_mgr.fname AS mgr_fname, e_mgr.lname AS mgr_lname
    FROM employee AS e INNER JOIN employee AS e_mgr
		ON e.superior_emp_id = e_mgr.emp_id;
        
/* NON-EQUI-JOINS */
SELECT e1.fname, e1.lname, 'VS' AS vs, e2.fname, e2.lname
	FROM employee AS e1 INNER JOIN employee AS e2
		ON e1.emp_id < e2.emp_id
	WHERE e1.title = 'Teller' AND e2.title = 'Teller';
    

/* EXERCISE 5-1 */
SELECT e.emp_id, e.fname, e.lname, b.name
	FROM employee AS e INNER JOIN branch As b
		ON e.assigned_branch_id = b.branch_id;
        
/* EXERCISE 5-2 */
SELECT a.account_id, c.fed_id, p.name
	FROM account AS a INNER JOIN customer As c
		ON a.cust_id = c.cust_id
        INNER JOIN product AS p
        ON a.product_cd = p.product_cd
        WHERE c.cust_type_cd = 'I';
        
/* EXERCISE 5-3 */
SELECT e1.emp_id, e1.fname, e1.lname
	FROM employee AS e1 INNER JOIN employee As e2
		ON e1.emp_id = e2.superior_emp_id
	WHERE e1.dept_id != e2.dept_id;
    





/* CHAPTER 6: WORKING WITH SETS */
DESC product;
DESC customer;
SELECT 1 num, 'abc' str
	UNION
SELECT 9 num, 'xyz' str;


/* UNION ALL: DOESN'T REMOVE DUPLICATES */
SELECT 'IND' AS type_cd, cust_id, lname AS name
	FROM individual
    UNION ALL
    SELECT 'BUS' AS type_cd, cust_id, name
    FROM business;
    
SELECT 'IND' AS type_cd, cust_id, lname AS name
	FROM individual
    UNION ALL
    SELECT 'BUS' AS type_cd, cust_id, name
    FROM business
    UNION ALL
    SELECT 'BUS' AS type_cd, cust_id, name
    FROM business;
    
 /* UNION: REMOVEs DUPLICATES */
SELECT 'IND' AS type_cd, cust_id, lname AS name
	FROM individual
    UNION
    SELECT 'BUS' AS type_cd, cust_id, name
    FROM business
    UNION
    SELECT 'BUS' AS type_cd, cust_id, name
    FROM business;
    
    
/* INTERSECT & EXCEPT: kann in MYSQL nicht genutzt werden; In ORACLE ja*/
/* SELECT emp_id
	FROM employee
    WHERE assigned_branch_id = 2
		AND (title = 'Teller' OR title = 'Head Teller')
	INTERSECT
    SELECT DISTINCT open_emp_id
    FROM account
    WHERE open_branch_id = 2;
  */
  
 /* IN MYSQL SIND KLAMMER NICHT MÖGLICH bei SET-OPERATOREN */
 /*
(SELECT cust_id
	FROM account
    WHERE product_cd IN ('SAV', 'MM')
    UNION ALL
    SELECT a.cust_id
		FROM account AS a INNER JOIN branch AS b
			ON a.open_branch_id = b.branch_id)
UNION
(SELECT cust_id
		FROM account
        WHERE avail_balance BETWEEN 500 AND 2500
        UNION
        SELECT cust_id
			FROM account 
            WHERE product_cd = 'CD'
            AND avail_balance < 1000);
*/

/* EXERCISE 6-2*/
SELECT fname, lname, 'CUST' AS status
	FROM individual
    UNION
    SELECT fname, lname, 'EMP' AS stautus
    FROM employee;
    
/* EXERCISE 6-3*/
SELECT fname, lname, 'CUST' AS status
	FROM individual
    UNION
    SELECT fname, lname, 'EMP' AS stautus
    FROM employee
    ORDER BY lname;







/* CHAPTER 7: Data Generation, Conversion and Manipulation */
CREATE TABLE string_tbl
	(char_fld CHAR(30),
    vchar_fld VARCHAR(30),
    text_fld TEXT
    );

    
INSERT INTO string_tbl (char_fld, vchar_fld, text_fld)
	VALUES ('This is char data',
			'This is varchar data',
			'This is text data');
            
/* ESCAPE */
UPDATE string_tbl
	SET text_fld='This string didn''t work, but it does now';
    
SELECT quote(text_fld)
	FROM string_tbl;
    
SELECT text_fld
	FROM string_tbl;
    
SELECT CHAR(148,149,150,151,152, 153,154,155);

SELECT concat('danke sch', CHAR(148),'n');

SELECT ascii('ä');
SELECT ascii('ö');


DELETE FROM string_tbl;
INSERT INTO string_tbl (char_fld, vchar_fld, text_fld)
	VALUES ('This string is 28 characters',
			'This string is 28 characters',
			'This string is 28 characters');


SELECT * FROM string_tbl;


SELECT length(char_fld) AS char_length,
	length(vchar_fld) AS vchar_length,
    length(text_fld) AS text_length
    FROM string_tbl;
    
SELECT cust_id, cust_type_cd, fed_id,
	fed_id REGEXP '.{3}-.{2}-.{4}' AS is_ss_no_format
	FROM customer;
    
SELECT insert('goodbye world', 9, 0, 'cruel ') AS string;
SELECT insert('goodbye world', 1, 7, 'cruel ') AS string;
SELECT substring('goodbye world', 9, 5);




SELECT ceil(72.4363), floor(72.8484846);
SELECT round(72.4363), round(72.8484846);
SELECT round(72.4363, 1), round(72.8484846, 2);
SELECT truncate(72.4363, 1), truncate(72.8484846, 3);
SELECT round(72.4363, -1), truncate(72.8484846, -1);

SELECT @@global.time_zone, @@session.time_zone;

UPDATE individual
SET birth_date = str_to_date('September 17, 2008', '%M %d, %Y')
WHERE cust_id = 9999;

SELECT current_date(), current_time(), current_timestamp();

SELECT DATE_ADD(current_date(), interval 5 day);
SELECT DATE_ADD(current_date(), interval 5 month);
SELECT DATE_ADD(current_date(), interval '3:27:11' HOUR_SECOND);
SELECT last_day('2008-09-17');
SELECT current_timestamp() AS current_est, convert_tz(current_timestamp(), 'US/Eastern', 'UTC') As current_utc;
SELECT DAYNAME('2008-09-18');
SELECT EXTRACT(YEAR from '2008-09-18 22:19:05');
SELECT datediff('2009-09-03 23:59:59', '2009-06-24 00:00:01');



/* EXERCISE 7-1 */
SELECT substring('Pluase find the substring in this string', 17,9);

/* EXERCISE 7-2 */
SELECT ABS(-25.76823), SIGN(-25.76823), round(-25.76823,2);

/* EXERCISE 7-3 */
SELECT EXTRACT(month from current_date());






/* CHAPTER 8: GROUPING AND AGGREGATES */
SELECT open_emp_id, COUNT(*) AS how_many
	FROM account
    GROUP BY open_emp_id;
    
SELECT open_emp_id, COUNT(*) AS how_many
	FROM account
    GROUP BY open_emp_id
    HAVING COUNT(*) > 4;
    
SELECT MAX(avail_balance) AS max_balance,
	MIN(avail_balance) AS min_balance,
    AVG(avail_balance) AS avg_balance,
    SUM(avail_balance) AS tot_balance,
    COUNT(*) AS num_accounts
	FROM account
    WHERE product_cd = 'CHK';
    
SELECT product_cd,
	MAX(avail_balance) AS max_balance,
	MIN(avail_balance) AS min_balance,
    AVG(avail_balance) AS avg_balance,
    SUM(avail_balance) AS tot_balance,
    COUNT(*) AS num_accounts
	FROM account
    GROUP BY product_cd;
    
SELECT COUNT(open_emp_id)
	FROM account;
    
SELECT COUNT(DISTINCT open_emp_id)
	FROM account;

SELECT MAX(pending_balance - avail_balance) AS max_uncleared
	FROM account;
    
    
CREATE TABLE number_tbl (val SMALLINT);
INSERT INTO number_tbl VALUES (1);
INSERT INTO number_tbl VALUES (3);
INSERT INTO number_tbl VALUES (5);
INSERT INTO number_tbl VALUES (NULL);

SELECT Count(*) num_rows,
	COUNT(val) num_vals,
    SUM(val) total,
    MAX(val) max_val,
    AVG(val) avg_val
    FROM number_tbl;
    
    
SELECT product_cd, SUM(avail_balance) AS prod_balance
	FROM account
    GROUP BY product_cd;
    
SELECT product_cd, open_branch_id,
	SUM(avail_balance) AS tot_balance
    FROM account
    GROUP BY product_cd, open_branch_id;
    
SELECT EXTRACT(YEAR FROM start_date) AS year,
	COUNT(*) AS how_many
    FROM employee
    GROUP BY EXTRACT(Year FROM start_date);
    
/* AUFSUMMIERUNG über die Gruppen */
/* Weitere Option WITH CUBE ist in MYSQL nicht möglich, aberin ORACLE */
SELECT product_cd, open_branch_id,
	SUM(avail_balance) AS tot_balance
    FROM account
    GROUP BY product_cd, open_branch_id WITH ROLLUP;
    
SELECT product_cd, SUM(avail_balance) AS prod_balance
    FROM account
    WHERE status = 'ACTIVE'
    GROUP BY product_cd
    HAVING SUM(avail_balance) >= 10000;

SELECT product_cd, SUM(avail_balance) AS prod_balance
    FROM account
    WHERE status = 'ACTIVE'
    GROUP BY product_cd
    HAVING MIN(avail_balance) >= 1000
		AND MAX(avail_balance) <= 10000;


/* EXERCISE 8-1 */
SELECT COUNT(*)
	FROM account;
    
/* EXERCISE 8-2 */
SELECT cust_id, COUNT(*) AS num_acc
	FROM account
    GROUP BY cust_id;

/* EXERCISE 8-3 */
SELECT cust_id, COUNT(*) AS num_acc
	FROM account
    GROUP BY cust_id
    HAVING COUNT(*) >= 2;
    
/* EXERCISE 8-4 */
SELECT product_cd, open_branch_id, SUM(avail_balance) AS tot_balance
	FROM account
    GROUP BY product_cd, open_branch_id
    HAVING COUNT(*) > 1
    ORDER BY tot_balance DESC;









/* CHAPTER 9: SUBQUERIES */

SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE account_id = (SELECT MAX(account_id) FROM account);
    
/* Noncorrelated subqueries */
SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE open_emp_id <> (SELECT e.emp_id
		FROM employee AS e INNER JOIN branch AS b
			ON e.assigned_branch_id = b.branch_id
		WHERE e.title = 'Head Teller' And b.city = 'Woburn');

SELECT branch_id, name, city
	FROM branch
    WHERE name IN ('Headquarters', 'Quincy Branch');

SELECT emp_id, fname, lname, title
	FROM employee
    WHERE emp_id IN (SELECT superior_emp_id FROM employee);
    
SELECT emp_id, fname, lname, title
	FROM employee
    WHERE emp_id NOT IN (SELECT superior_emp_id FROM employee WHERE superior_emp_id IS NOT NULL);
    
SELECT emp_id, fname, lname, title
	FROM employee
    WHERE emp_id <> ALL (SELECT superior_emp_id FROM employee WHERE superior_emp_id IS NOT NULL);



/* der Unterschied zwischen ALL und ANY */
SELECT account_id, cust_id, product_cd, avail_balance
	FROM account
    WHERE avail_balance < ALL (SELECT a.avail_balance
		FROM account As a INNER JOIN individual As i
			ON a.cust_id = i.cust_id
        WHERE i.fname = 'Frank' AND i.lname = 'Tucker');
        
SELECT account_id, cust_id, product_cd, avail_balance
	FROM account
    WHERE avail_balance < ANY (SELECT a.avail_balance
		FROM account As a INNER JOIN individual As i
			ON a.cust_id = i.cust_id
        WHERE i.fname = 'Frank' AND i.lname = 'Tucker');


SELECT account_id, product_cd, cust_id
	FROM account
    WHERE open_branch_id = (SELECT branch_id
		FROM branch
        WHERE name = 'Woburn Branch')
        AND open_emp_id IN (SELECT emp_id
        FROM employee
        WHERE title = 'Teller' OR title = 'HEAD Teller');
        
SELECT account_id, product_cd, cust_id
	FROM account
    WHERE (open_branch_id, open_emp_id) IN
		(SELECT b.branch_id, e.emp_id
			FROM branch As b INNER JOIN employee As e
				ON b.branch_id = e.assigned_branch_id
            WHERE b.name = 'Woburn Branch'
            AND (e.title = 'Teller' OR e.title = 'HEAD Teller'));     


/* Correlated Subqueries */
SELECT c.cust_id, c.cust_type_cd, c.city
	FROM customer c
	WHERE 2 = (SELECT COUNT(*)
		FROM account a
        WHERE a.cust_id = c.cust_id);
        
SELECT c.cust_id, c.cust_type_cd, c.city
	FROM customer c
	WHERE (SELECT SUM(a.avail_balance)
			FROM account a
			WHERE a.cust_id = c.cust_id)
        BETWEEN 5000 AND 10000;



/* Convention is to specify either SELCT 1 ot SELECT * when using exists */
SELECT a.account_id, a.product_cd, a.cust_id, a.avail_balance
	FROM account a
    WHERE EXISTS (SELECT 1
		FROM transaction t
        WHERE t.account_id = a.account_id
			AND t.txn_date = '2008-09-22');

SELECT a.account_id, a.product_cd, a.cust_id, a.avail_balance
	FROM account a
    WHERE NOT EXISTS (SELECT 1
		FROM business b
        WHERE b.cust_id = a.cust_id);

UPDATE account a
SET a.last_activity_date = 
	(SELECT MAX(t.txn_date)
		FROM transaction t 
        WHERE t.account_id = a.account_id)
	WHERE EXISTS (SELECT 1
		FROM transaction t
        WHERE t.account_id = a.account_id);
        
DELETE FROM department
	WHERE NOT EXISTS (SELECT 1
		FROM employee
        WHERE employee.dept_id = department.dept_id);
        
SELECT d.dept_id, d.name, e_cnt.how_many AS num_employees
	FROM department AS d INNER JOIN
		(SELECT dept_id, COUNT(*) AS how_many
			FROM employee
            GROUP BY dept_id) AS e_cnt
            ON d.dept_id = e_cnt.dept_id;


/* Group definitions */
SELECT 'Small Fry' name, 0 low_limit, 4999.9 high_limit
UNION ALL
SELECT 'Average Joes' name, 5000 low_limit, 9999.99 high_limit
UNION ALL
SELECT 'Heavy Hitters' name, 1000 low_limit, 9999999.99 high_limit;


SELECT gr.name, COUNT(*) num_customers
	FROM 
		(SELECT SUM(a.avail_balance) cust_balance 
			FROM account a INNER JOIN product p 
				ON a.product_cd = p.product_cd
			WHERE p.product_type_cd = 'ACCOUNT'
            GROUP BY a.cust_id) cust_rollup
            INNER JOIN 
            (SELECT 'Small Fry' name, 0 low_limit, 4999.9 high_limit
			UNION ALL 
			SELECT 'Average Joes' name, 5000 low_limit, 9999.99 high_limit
			UNION ALL 
			SELECT 'Heavy Hitters' name, 1000 low_limit, 9999999.99 high_limit) gr 
            ON cust_rollup.cust_balance  BETWEEN gr.low_limit AND gr.high_limit
		GROUP BY gr.name;


SELECT p.name AS product,
		b.name AS branch,
		CONCAT(e.fname, ' ', e.lname) AS name,
        SUM(a.avail_balance) AS tot_deposits
	FROM account a INNER JOIN employee e
		ON a.open_emp_id = e.emp_id
        INNER JOIN branch b
        ON a.open_branch_id = b.branch_id
        INNER JOIN product p
        ON a.product_cd = p.product_cd
	WHERE p.product_type_cd = 'ACCOUNT'
    GROUP BY p.name, b.name, e.fname, e.lname
    ORDER BY 1, 2;

/* SAME QUERY BUT BETTER */
SELECT p.name AS product,
		b.name AS branch,
		CONCAT(e.fname, ' ', e.lname) AS name,
        account_groups.tot_deposits
	FROM 
		(SELECT product_cd, 
			open_branch_id AS branch_id,
			open_emp_id AS emp_id,
            SUM(avail_balance) AS tot_deposits
            FROM account
            GROUP BY product_cd, open_branch_id, open_emp_id) account_groups
		INNER JOIN employee e ON e.emp_id = account_groups.emp_id
        INNER JOIN branch b ON b.branch_id = account_groups.branch_id
        INNER JOIN product p ON p.product_cd = account_groups.product_cd
	WHERE p.product_type_cd = 'ACCOUNT';
            
        
SELECT open_emp_id, COUNT(*) how_many
	FROM account
    GROUP BY open_emp_id
    HAVING COUNT(*) = (SELECT MAX(emp_cnt.how_many)
		FROM (SELECT COUNT(*) how_many
			FROM account
            GROUP BY open_emp_id) emp_cnt
    );

SELECT
	(SELECT p.name FROM product p
		WHERE p.product_cd = a.product_cd
        AND p.product_type_cd = 'ACCOUNT') product,
    (SELECT b.name FROM branch b
		WHERE b.branch_id = a.open_branch_id) branch,
    (SELECT CONCAT(e.fname, ' ', e.lname) FROM employee e
		WHERE e.emp_id = a.open_emp_id) name,
    SUM(a.avail_balance) tot_deposits
	FROM account a
    GROUP BY a.product_cd, a.open_branch_id, a.open_emp_id
    ORDER BY 1,2;

/* ENTFERNUNG VON NULL-ZEILEN */
SELECT all_prods.product, all_prods.branch, all_prods.name, all_prods.tot_deposits
	FROM (
		SELECT
		(SELECT p.name FROM product p
			WHERE p.product_cd = a.product_cd
			AND p.product_type_cd = 'ACCOUNT') product,
		(SELECT b.name FROM branch b
			WHERE b.branch_id = a.open_branch_id) branch,
		(SELECT CONCAT(e.fname, ' ', e.lname) FROM employee e
			WHERE e.emp_id = a.open_emp_id) name,
		SUM(a.avail_balance) tot_deposits
		FROM account a
		GROUP BY a.product_cd, a.open_branch_id, a.open_emp_id
		ORDER BY 1,2
		) all_prods
    WHERE all_prods.product IS NOT NULL
    ORDER BY 1,2;

SELECT emp.emp_id, CONCAT(emp.fname, ' ', emp.lname) emp_name,
	(SELECT CONCAT(boss.fname, ' ', boss.lname)
		FROM employee boss
        WHERE boss.emp_id = emp.superior_emp_id) boss_name
	FROM employee emp
    WHERE emp.superior_emp_id IS NOT NULL
    ORDER BY (SELECT boss.lname FROM employee boss
		WHERE boss.emp_id = emp.superior_emp_id), emp.lname;
        

INSERT INTO account 
	(account_id, product_cd, cust_id, open_date, last_activity_date. status, open_branch_id, open_emp_id, avail_balance, pending_balance)
    VALUES(NULL,
		(SELECT product_cd FROM product WHERE name = 'savings account'),
        (SELECT cust_id FROM customer WHERE fed_id = '555-55-5555'),
        '2008-09-25',
        '2008-09-25',
        'ACTIVE',
        (SELECT branch_id FROM branch WHERE name = 'Quincy Branch'),
        (SELECT emp_id FROM employee WHERE lname = 'Portman' AND fname = 'Frank'),
        0,
        0
        );


/* EXERCISE 9-1 */
SELECT account_id, product_cd, cust_id, avail_balance
	FROM account
    WHERE product_cd IN (SELECT product_cd
		FROM product
        WHERE product_type_cd = 'LOAN');
        
/* EXERCISE 9-2 */
SELECT a.account_id, a.product_cd, a.cust_id, a.avail_balance
	FROM account a 
    WHERE EXISTS (SELECT 1
		FROM product p 
        WHERE p.product_cd = a.product_cd
			AND  p.product_type_cd = 'LOAN');


/* EXERCISE 9-3 */
SELECT e.emp_id, e.fname, e.lname, levels.name
	FROM employee e INNER JOIN
    (
	SELECT 'trainee' name, '2004-01-01' start_dt, '2005-12-31' end_dt
	UNION ALL
	SELECT 'worker' name, '2002-01-01' start_dt, '2003-12-31' end_dt
	UNION ALL 
	SELECT 'mentor' name, '2000-01-01' start_dt, '2001-12-31' end_dt
    ) levels
    ON e.start_date BETWEEN start_dt AND end_dt;



/* EXERCISE 9-4 */
SELECT e.emp_id, e.fname, e.lname,
	(SELECT d.name FROM department d WHERE e.dept_id = d.dept_id) dept_name,
   	(SELECT b.name FROM branch b WHERE e.assigned_branch_id = b.branch_id) branch_name
	FROM employee e;




SELECT CONCAT('ALERT! : Account #', a.account_id, 'Has Incorrect Balance!')
	FROM account AS a
    WHERE (a.avail_balance, a.pendng_balance) <>
		(SELECT SUM(), SUM()
        FROM transaction t
        WHERE t.account_id = a.account_id);






/*CHAPTER 10:  JOINS REVISITED */
SELECT account_id, cust_id FROM account;
SELECT cust_id FROM customer;        
SELECT a.account_id, c.cust_id
	FROM account a INNER JOIN customer c
    ON a.cust_id = c.cust_id;
    
SELECT a.account_id, b.cust_id, b.name
	FROM account a INNER JOIN business b
    ON a.cust_id = b.cust_id;
SELECT cust_id, name FROM business;

SELECT a.account_id, a.cust_id, b.name
	FROM account a LEFT OUTER JOIN business b
    ON a.cust_id = b.cust_id;

SELECT a.account_id, a.cust_id, i.fname, i.lname
	FROM account a LEFT OUTER JOIN individual i
    ON a.cust_id = i.cust_id;

SELECT c.cust_id, b.name
	FROM customer c LEFT OUTER JOIN business b
    ON c.cust_id = b.cust_id;

SELECT c.cust_id, b.name
	FROM customer c RIGHT OUTER JOIN business b
    ON c.cust_id = b.cust_id;

SELECT a.account_id, a.product_cd,
	CONCAT(i.fname, '', i.lname) person_name,
    b.name business_name
    FROM account a LEFT OUTER JOIN individual i
    ON a.cust_id = i.cust_id
    LEFT OUTER JOIN business b 
    ON a.cust_id = b.cust_id;
    
    
SELECT account_ind.account_id, account_ind.product_cd,
	account_ind.person_name,
    b.name business_name
FROM 
    (SELECT a.account_id, a.product_cd, a.cust_id,
	CONCAT(i.fname, '', i.lname) person_name
    FROM account a LEFT OUTER JOIN individual i
    ON a.cust_id = i.cust_id) account_ind
    LEFT OUTER JOIN business b 
    ON account_ind.cust_id = b.cust_id;
    
    
SELECT e.fname, e.lname,
	e_mgr.fname mgr_fname, e_mgr.lname mgr_lname
    FROM employee e INNER JOIN employee e_mgr
    ON e.superior_emp_id = e_mgr.emp_id;
    
    
SELECT e.fname, e.lname,
	e_mgr.fname mgr_fname, e_mgr.lname mgr_lname
    FROM employee e LEFT OUTER JOIN employee e_mgr
    ON e.superior_emp_id = e_mgr.emp_id;
    
SELECT e.fname, e.lname,
	e_mgr.fname mgr_fname, e_mgr.lname mgr_lname
    FROM employee e LEFT OUTER JOIN employee e_mgr
    ON e.superior_emp_id = e_mgr.emp_id;
    
    
SELECT pt.name, p.product_cd, p.name
	FROM product p CROSS JOIN product_type pt;
    
    
    

SELECT date_add('2008-01-01', INTERVAL(ones.num + tens.num + hundreds.num) DAY) dt
	FROM
(SELECT 0 num UNION ALL
SELECT 1 numm UNION ALL
SELECT 2 numm UNION ALL
SELECT 3 numm UNION ALL
SELECT 4 numm UNION ALL
SELECT 5 numm UNION ALL
SELECT 6 numm UNION ALL
SELECT 7 numm UNION ALL
SELECT 8 numm UNION ALL
SELECT 9 numm) ones
CROSS JOIN
(SELECT 0 num UNION ALL
SELECT 10 numm UNION ALL
SELECT 20 numm UNION ALL
SELECT 30 numm UNION ALL
SELECT 40 numm UNION ALL
SELECT 50 numm UNION ALL
SELECT 60 numm UNION ALL
SELECT 70 numm UNION ALL
SELECT 80 numm UNION ALL
SELECT 90 numm) tens
CROSS JOIN
(SELECT 0 num UNION ALL
SELECT 100 numm UNION ALL
SELECT 200 numm UNION ALL
SELECT 300 numm) hundreds
WHERE date_add('2008-01-01', interval(ones.num + tens.num + hundreds.num) DAY) < '2009-01-01' ORDER BY 1;





SELECT days.dt, COUNT(t.txn_id)
	FROM transaction t RIGHT OUTER JOIN
    (SELECT date_add('2008-01-01', INTERVAL(ones.num + tens.num + hundreds.num) DAY) dt
	FROM
	(SELECT 0 num UNION ALL
	SELECT 1 num UNION ALL
	SELECT 2 num UNION ALL
	SELECT 3 num UNION ALL
	SELECT 4 num UNION ALL
	SELECT 5 num UNION ALL
	SELECT 6 num UNION ALL
	SELECT 7 num UNION ALL
	SELECT 8 num UNION ALL
	SELECT 9 num) ones
	CROSS JOIN
	(SELECT 0 num UNION ALL
	SELECT 10 num UNION ALL
	SELECT 20 num UNION ALL
	SELECT 30 num UNION ALL
	SELECT 40 num UNION ALL
	SELECT 50 num UNION ALL
	SELECT 60 num UNION ALL
	SELECT 70 num UNION ALL
	SELECT 80 num UNION ALL
	SELECT 90 num) tens
	CROSS JOIN
	(SELECT 0 num UNION ALL
	SELECT 100 num UNION ALL
	SELECT 200 num UNION ALL
	SELECT 300 num) hundreds
	WHERE date_add('2008-01-01', interval(ones.num + tens.num + hundreds.num) DAY) < '2009-01-01') days
    ON days.dt = t.txn_date
    GROUP BY days.dt
    ORDER BY 1;



/* Natutal Join */
SELECT a.account_id, a.cust_id, c.cust_type_cd, c.fed_id
	FROM account a NATURAL JOIN customer c;
    
SELECT a.account_id, a.cust_id, a.open_branch_id, b.name
	FROM account a NATURAL JOIN branch b;
    
    
/* EXERCISE 10-1 */
SELECT p.product_cd, a.account_id, a.cust_id, a.avail_balance 
	FROM product p LEFT OUTER JOIN account a
    ON p.product_cd = a.product_cd
    ORDER BY 1;
    
/* EXERCISE 10-2 */
SELECT p.product_cd, a.account_id, a.cust_id, a.avail_balance 
	FROM account a RIGHT OUTER JOIN product p
    ON a.product_cd = p.product_cd
    ORDER BY 1;
    
/* EXERCISE 10-3 */
SELECT a.account_id, a.product_cd, i.fname, i.lname, b.name 
	FROM account a LEFT OUTER JOIN individual i
    ON a.cust_id = i.cust_id
    LEFT OUTER JOIN business b
    ON a.cust_id = b.cust_id;
    
SELECT a.account_id, a.product_cd, i.fname, i.lname, b.name 
	FROM account a LEFT OUTER JOIN business b
    ON a.cust_id = b.cust_id
    LEFT OUTER JOIN individual i
    ON a.cust_id = i.cust_id;
    
/* EXERCISE 10-4 */
SELECT ones.num + tens.num FROM
(SELECT 1 num UNION ALL
SELECT 2 num UNION ALL
SELECT 3 num UNION ALL
SELECT 4 num UNION ALL
SELECT 5 num UNION ALL
SELECT 6 num UNION ALL
SELECT 7 num UNION ALL
SELECT 8 num UNION ALL
SELECT 9 num UNION ALL
SELECT 10 num) ones
CROSS JOIN
(SELECT 0 num UNION ALL
SELECT 10 num UNION ALL
SELECT 20 num UNION ALL
SELECT 30 num UNION ALL
SELECT 40 num UNION ALL
SELECT 50 num UNION ALL
SELECT 60 num UNION ALL
SELECT 70 num UNION ALL
SELECT 80 num UNION ALL
SELECT 90 num) tens;


SELECT ones.num + tens.num + 1 FROM
(SELECT 0 num UNION ALL
SELECT 1 num UNION ALL
SELECT 2 num UNION ALL
SELECT 3 num UNION ALL
SELECT 4 num UNION ALL
SELECT 5 num UNION ALL
SELECT 6 num UNION ALL
SELECT 7 num UNION ALL
SELECT 8 num UNION ALL
SELECT 9 num) ones
CROSS JOIN
(SELECT 0 num UNION ALL
SELECT 10 num UNION ALL
SELECT 20 num UNION ALL
SELECT 30 num UNION ALL
SELECT 40 num UNION ALL
SELECT 50 num UNION ALL
SELECT 60 num UNION ALL
SELECT 70 num UNION ALL
SELECT 80 num UNION ALL
SELECT 90 num) tens;








/* CHAPTER 11: CONDITIONAL LOGIC */
SELECT c.cust_id, c.fed_id, c.cust_type_cd,
	CONCAT(i.fname, ' ', i.lname) indiv_name,
    b.name business_name
    FROM customer c LEFT OUTER JOIN individual i
    ON c.cust_id = i.cust_id
    LEFT OUTER JOIN business b
    ON c.cust_id = b.cust_id;
    
SELECT c.cust_id, c.fed_id, c.cust_type_cd,
	CASE
		WHEN c.cust_type_cd = 'I'
			THEN CONCAT(i.fname, ' ', i.lname)
		WHEN c.cust_type_cd = 'B'
			THEN b.name
		ELSE 'Unknown'
        END name
	FROM customer c LEFT OUTER JOIN individual i
    ON c.cust_id = i.cust_id
    LEFT OUTER JOIN business b
    ON c.cust_id = b.cust_id;
    
    
SELECT c.cust_id, c.fed_id, c.cust_type_cd,
	CASE
		WHEN c.cust_type_cd = 'I'
			THEN 
				(SELECT CONCAT(i.fname, ' ', i.lname)
					FROM individual i
                    WHERE i.cust_id = c.cust_id)
		WHEN c.cust_type_cd = 'B'
			THEN 
				(SELECT b.name
					FROM business b
                    WHERE b.cust_id = c.cust_id)
		ELSE 'Unknown'
        END name
	FROM customer c;
    
    
    
    SELECT YEAR(open_date) year, COUNT(*) how_many
		FROM account
        WHERE open_date > '1999-12-31'
			AND open_date < '2006-01-01'
		GROUP BY YEAR(open_date);
        
	
SELECT 
			SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2000 THEN 1
				ELSE 0
            END) year_2000,
			SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2001 THEN 1
				ELSE 0
            END) year_2001,
            SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2002 THEN 1
				ELSE 0
            END) year_2002,
            SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2003 THEN 1
				ELSE 0
            END) year_2003,
            SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2004 THEN 1
				ELSE 0
            END) year_2004,
            SUM(CASE
				WHEN EXTRACT(YEAR FROM open_date) = 2005 THEN 1
				ELSE 0
            END) year_2005
            FROM account
            WHERE open_date > '1999-12-31' AND open_date < '2006-0101';
            
            
            
SELECT CONCAT('ALERT! : Account #', a.account_id, 'Has Incorrect Balance!')
	FROM account AS a
    WHERE (a.avail_balance, a.pendng_balance) <>
		(SELECT SUM(CASE
						WHEN t.funds_avail_date > current_timestamp() THEN 0
						WHEN t.txn_type_cd = 'DBT' THEN t.amount * -1
                        ELSE t.amount
						END), 
				SUM(CASE
						WHEN t.txn_type_cd = 'DBT' THEN t.amount * -1
                        ELSE t.amount
                        END)
        FROM transaction t
        WHERE t.account_id = a.account_id);


/* FLAG */
SELECT c.cust_id, c.fed_id, c.cust_type_cd,
	CASE
		WHEN EXISTS (SELECT 1 FROM account a 
			WHERE a.cust_id = c.cust_id
				AND a.product_cd = 'CHK') THEN 'Y'
		ELSE 'N'
	END has_checking,
    CASE
		WHEN EXISTS (SELECT 1 FROM account a 
			WHERE a.cust_id = c.cust_id
				AND a.product_cd = 'SAV') THEN 'Y'
		ELSE 'N'
	END has_savings
FROM customer c;



SELECT c.cust_id, c.fed_id, c.cust_type_cd,
	CASE (SELECT COUNT(*) FROM account a
		WHERE a.cust_id = c.cust_id)
        WHEN 0 THEN 'NONE'
        WHEN 1 THEN '1'
        WHEN 2 THEn '2'
        ELSE '3+'
	END num_accounts
FROM customer c;


/* Division by Zero errors */
SELECT a.cust_id, a.product_cd, a.avail_balance / 
	CASE
		WHEN prod_tots.tot_balance = 0 THEN 1
		ELSE prod_tots.tot_balance
	END percent_of_total
    FROM account a INNER JOIN 
		(SELECT a.product_cd, SUM(a.avail_balance) tot_balance
			FROM account a
            GROUP BY a.product_cd) prod_tots
            ON a.product_cd = prod_tots.product_cd;


/* Handling NULL Values */
SELECT emp_id, fname, lname,
	CASE
		WHEN title is NULL THEN 'Unknown'
        ELSE title
	END title
FROM employee;


/*
SELECT <some calculation> + 
	CASE
		WHEN avail_balance IS NULL THEN 0
		ELSE avail_balance
	END
    + <rest of calculation>
*/


/* EXERCISE 11-1 */
SELECT emp_id,
	CASE 
		WHEN title LIKE '%President' OR title = 'Loan Manager' OR title = 'Treasurer' THEN 'Management'
        WHEN title LIKE '%Teller' OR title = 'Operations Manager' THEN 'Operations'
        ELSE 'Unknown'
	END
FROM employee;

/* EXERCISE 11-2 */
SELECT 
	SUM(CASE
		WHEN open_branch_id = 1 THEN 1
        ELSE 0
        END) branch_1,
	SUM(CASE
		WHEN open_branch_id = 2 THEN 1
        ELSE 0
        END) branch_2,
	SUM(CASE
		WHEN open_branch_id = 3 THEN 1
        ELSE 0
        END) branch_3,
	SUM(CASE
		WHEN open_branch_id = 4 THEN 1
        ELSE 0
        END) branch_4
FROM account;












/* CHAPTER 12 */
/* EXERCISE 12-1 */

START TRANSACTION;

SELECT i.cust_id, 
	(SELECT a.account_id  FROM account a
	 WHERE a.cust_id = i.cust_id AND a.product_cd = 'MM') mm_id,
    (SELECT a.account_id  FROM account a
	 WHERE a.cust_id = i.cust_id AND a.product_cd = 'chk') chk_id
	INTO @cst_id, @mm_id, @chk_id
    FROM individual i
    WHERE i.fname = 'Frank' AND i.lname = 'Tucker';

INSERT INTO transaction (txn_id, txn_date, account_id, txn_type_cd, amount)
	VALUES (NULL, now(), @mm_id, 'CDT', 50);
    
INSERT INTO transaction (txn_id, txn_date, account_id, txn_type_cd, amount)
	VALUES (NULL, now(), @chk_id, 'DBT', 50);

UPDATE account 
SET last_activity_date = now(),
	avail_balance = aval_balance - 50
WHERE account_id = @mm_id;

UPDATE account 
SET last_activity_date = now(),
	avail_balance = aval_balance + 50
WHERE account_id = @chk_id;

COMMIT;









/* CHAPTER 13 */
/* INDEXES AND CONSTRAINTS */

CREATE INDEX dept_name_idx ON department (name);
SHOW INDEX FROM department;
ALTER TABLE employee ADD INDEX emp_names (lname, fname);
SHOW INDEX FROM employee;

EXPLAIN SELECT cust_id, SUM(avail_balance) tot_bal
FROM account
WHERE cust_id IN (1,5,9,11)
GROUP BY cust_id;




ALTER TABLE account
ADD INDEX acc_bal_idv (cust_id, avail_balance);

EXPLAIN SELECT cust_id, SUM(avail_balance) tot_bal
FROM account
WHERE cust_id IN (1,5,9,11)
GROUP BY cust_id;

SELECT cust_id
FROM account;


/* Constraint Creation */
CREATE Table product
	(product_cd VARCHAR(10) NOT NULL,
	name VARCHAR(50) NOT NULL,
    product_type_cd VARCHAR(10) NOT NULL,
    date_offered DATE,
    date_retired DATE,
		CONSTRAINT fk_product_type_cd FOREIGN KEY (product_type_cd)
			REFERENCES product_type (product_type_cd),
		CONSTRAINT pk_product  PRIMARY KEY (product_cd)
        );
        
        
/*Cascading Constraints */
SELECT product_type_cd, name FROM product_type;
SELECT product_type_cd, product_cd, name FROM product ORDER BY product_type_cd;
/* Das funktioniert nicht */
UPDATE product
	SET product_type_cd = 'XYZ'
    WHERE product_type_cd = 'LOAN';
    
/* Das funktioniert */
SHOW INDEX FROM product;
ALTER TABLE product
	DROP FOREIGN KEY fk_product_type_cd;
ALTER TABLE product
	ADD CONSTRAINT fk_product_type_cd FOREIGN KEY (product_type_cd)
    REFERENCES product_type (product_type_cd)
    ON UPDATE CASCADE;
UPDATE product_type
	SET product_type_cd = 'XYZ'
    WHERE product_type_cd = 'LOAN';


/* EXERCISE 13-1 */
SELECT * FROM account limit 10;
ALTER TABLE account
	ADD CONSTRAINT account_unq1 UNIQUE (cust_id, product_cd);
SHOW INDEX FROM account;


/* EXERCISE 13-2 */
SELECT * FROM transaction limit 10;
CREATE INDEX txn_idx01 ON transaction (txn_date, amount);
SHOW INDEX FROM transaction;

SELECT txn_date, account_id, txn_type_cd, amount
	FROM transaction
    WHERE txn_date > cast('2008-12-31 23:59:59' as datetime);
 
SELECT txn_date, account_id, txn_type_cd, amount
	FROM transaction
    WHERE txn_date > cast('2008-12-31 23:59:59' as datetime)
		AND amount < 1000;











/* CHAPTER 14 */
CREATE VIEW customer_vw
	(cust_id,
    fed_id,
    cust_type_cd,
    address,
    city,
    state,
    zipcode
    )
    AS
    SELECT cust_id, 
    concat('ends in ', substr(fed_id, 8, 4)) fed_id,
        cust_type_cd,
    address,
    city,
    state,
    postal_code
    FROM customer;
    
SELECT cust_id, fed_id, cust_type_cd FROM customer_vw;

DESCRIBE customer_vw;

SELECT cust_type_cd, count(*)
	FROM customer_vw
    WHERE state = 'MA'
    GROUP BY cust_type_cd
    ORDER bY 1;


SELECT cst.cust_id, cst.fed_id, bus.name
	FROM customer_vw cst
    INNER JOIN business bus
    ON cst.cust_id = bus.cust_id;
    
CREATE VIEW business_customer_vw
	(cust_id,
    fed_id,
    cust_type_cd,
    address,
    city,
    state,
    zipcode
    )
    AS
    SELECT cust_id, 
    concat('ends in ', substr(fed_id, 8, 4)) fed_id,
        cust_type_cd,
    address,
    city,
    state,
    postal_code
    FROM customer
    WHERE cust_type_cd = 'B';

DESCRIBE business_customer_vw;




CREATE VIEW customer_totals_vw
	(cust_id,
    cust_type_cd,
    cust_name,
    num_accounts,
    tot_deposits
    )
    AS
    SELECT cst.cust_id, cst.cust_type_cd,
    CASE
		WHEN cst.cust_type_cd = 'B' THEN
        (SELECT bus.name FROM business bus WHERE bus.cust_id = cst.cust_id)
        ELSE
        (SELECT concat(ind.fname, ' ', ind.lname)
			FROM individual ind WHERE ind.cust_id = cst.cust_id)
        END cust_name,
        sum(CASE WHEN act.status = 'ACTIVE' THEN 1 ELSE 0 END) tot_active_accounts,
        sum(CASE WHEN act.status = 'ACTIVE' THEN act.avail_balance ELSE 0 END) tot_balance
	FROM customer cst
    INNER JOIN account act
    ON act.cust_id = cst.cust_id
    GROUP BY cst.cust_id, cst.cust_type_cd;

CREATE TABLE customer_totals
AS
SELECT * FROM customer_totals_vw;

CREATE OR REPLACE VIEW customer_totals_vw
	(cust_id,
    cust_type_cd,
    cust_name,
    num_accounts,
    tot_deposits
    )
    AS
    SELECT cust_id, cust_type_cd, cust_name, num_accounts, tot_deposits
    FROM customer_totals;
    
    
    
    /* Hiding complexity */
CREATE VIEW branch_activity_vw
		(branch_name,
        city,
        state,
        num_employees,
        num_active_accounts,
        tot_transactions
        )
        AS
        SELECT br.name, br.city, br.state,
			(SELECT count(*)
            FROM employee emp
            WHERE emp.assigned_branch_id = br.branch_id) num_emps,
			(SELECT count(*)
            FROM account acnt
            WHERE acnt.status = 'ACTIVE' AND acnt.open_branch_id = br.branch_id) num_account,
			(SELECT count(*)
            FROM transaction tnx
            WHERE tnx.execution_branch_id = br.branch_id) num_tnxs
		FROM branch br;
	
SELECT * FROM branch_activity_vw;


/* Updatable Views */
CREATE VIEW customer_vw
	(cust_id,
    fed_id,
    cust_type_cd,
    address,
    city,
    state,
    zipcode
    )
    AS
    SELECT cust_id, 
    concat('ends in ', substr(fed_id, 8, 4)) fed_id,
        cust_type_cd,
    address,
    city,
    state,
    postal_code
    FROM customer;
    
    UPDATE customer_vw
    SET city = 'Wooburn'
    WHERE city = 'Woburn';
    
SELECT DISTINCT city FROM customer;



CREATE VIEW business_customer_vw_2
	(cust_id,
    fed_id,
    address,
    city,
    state,
    postal_code,
    business_name,
    state_id,
    incorp_date
    )
    AS
    SELECT cst.cust_id,
	cst.fed_id,
	cst.address,
    cst.city,
    cst.state,
    cst.postal_code,
    bsn.name,
    bsn.state_id,
    bsn.incorp_date
    FROM customer cst INNER JOIN business bsn
    ON cst.cust_id = bsn.cust_id
    WHERE cust_type_cd = 'B';

DESCRIBE business_customer_vw_2;

UPDATE business_customer_vw_2
SET postal_code = '99999'
WHERE cust_id = 10;

UPDATE business_customer_vw_2
SET incorp_date = '2008-11-17'
WHERE cust_id = 10;



/* Exercise 14-1 */
CREATE VIEW supervisor_emp
	(supervisor_name,
	employee_name)
	AS
    SELECT 
	concat(spr.fname, ' ', spr.lname),
    concat(emp.fname, ' ', emp.lname)
    FROM employee emp LEFT OUTER JOIN employee spr
    ON emp.superior_emp_id = spr.emp_id;

SELECT * FROM supervisor_emp;


/* Exercise 14-2 */
CREATE VIEW branch_summary_vw
	(name,
    city,
    tot_balance)
    AS
    SELECT
    br.name, br.city,
    SUM(acnt.avail_balance)
    FROM branch br INNER JOIN account acnt
    ON br.branch_id = acnt.open_branch_id
    GROUP BY br.name, br.city;
    
SELECT * FROM branch_summary_vw;