
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

