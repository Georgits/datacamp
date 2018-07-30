library(stringr)
library(rebus)
library(dplyr)
library(babynames)
library(stringi)


# 1. Chapter 1: String basics -----
  # Quotes ----
  # Define line1
  line1 <- "The table was a large one, but the three were all crowded together at one corner of it:"
  
  # Define line2
  line2 <- '"No room! No room!" they cried out when they saw Alice coming.'
  
  # Define line3
  line3 <- "\"There's plenty of room!\" said Alice indignantly, and she sat down in a large arm-chair at one end of the table."

  
  
  
  # What you see isn't always what you have ----
  # Putting lines in a vector
  lines <- c(line1, line2, line3)
  
  # Print lines
  print(lines)
  
  # Use writeLines() on lines
  writeLines(lines)
  
  # Write lines with a space separator
  writeLines(lines, sep = " ")
  
  # Use writeLines() on the string "hello\n\U1F30D"
  writeLines("hello\n\U1F30D")
  
  
  # Escape sequences ----
  # Should display: To have a \ you need \\
  writeLines("To have a \\ you need \\\\")
  
  # Should display: 
  # This is a really 
  # really really 
  # long string
  writeLines("This is a really \n really really \n long string")
  
  # Use writeLines() with 
  # "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"
  
  # you just said "Hello World" in Hindi!
  print("\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e")
  # georgian
  print("\u10D9\u10D0\u10E0\u10D2\u10D8")
  
  
  
  
  # Using format() with numbers ----
  # Some vectors of numbers
  percent_change  <- c(4, -1.91, 3.00, -5.002)
  income <-  c(72.19, 1030.18, 10291.93, 1189192.18)
  p_values <- c(0.12, 0.98, 0.0000191, 0.00000000002)
  
  # Format c(0.0011, 0.011, 1) with digits = 1
  format(c(0.0011, 0.011, 1), digits = 1)
  
  # Format c(1.0011, 2.011, 1) with digits = 1
  format(c(1.0011, 2.011, 1), digits = 1)
  
  # Format percent_change to one place after the decimal point
  format(percent_change, digits = 2)
  
  # Format income to whole numbers
  format(income, digits = 2)
  
  # Format p_values in fixed format
  format(p_values, scientific = FALSE)
  
  
  # Controlling other aspects of the string ----
  formatted_income <- format(income, digits = 2)
  
  # Print formatted_income
  print(formatted_income)
  
  # Call writeLines() on the formatted income
  writeLines(formatted_income)
  
  # Define trimmed_income
  trimmed_income <- format(income, digits = 2, trim = TRUE)
  
  # Call writeLines() on the trimmed_income
  writeLines(trimmed_income)
  
  # Define pretty_income
  pretty_income <- format(income, digits = 2, big.mark = ",")

  # Call writeLines() on the pretty_income
  writeLines(pretty_income)

  
  
  # formatC() ----
  # From the format() exercise
  x <- c(0.0011, 0.011, 1)
  y <- c(1.0011, 2.011, 1)
  
  # formatC() on x with format = "f", digits = 1
  formatC(x, format = "f", digits = 1)
  
  # formatC() on y with format = "f", digits = 1
  formatC(y, format = "f", digits = 1)
  
  # Format percent_change to one place after the decimal point; "f" for fixed; "e" for scientific
  formatC(percent_change, format = "f", digits = 1)
  
  # percent_change with flag = "+"
  formatC(percent_change, format = "f", digits = 1, flag = "+")
  
  # Format p_values using format = "g" and digits = 2; "g" for fixed unless scientific saves space
  formatC(p_values, format = "g", digits = 2)
  
  
  
  # Annotation of numbers ----
  pretty_income <- formatC(income, digits = 0, big.mark = ",", format = "f")
  pretty_percent <- formatC(percent_change, digits = 1, format = "f")
  years <- c(2010, 2011, 2012, 2013)
  
  # Add $ to pretty_income
  paste("$", pretty_income, sep = "")
  
  # Add % to pretty_percent
  paste(pretty_percent, "%", sep = "")
  
  # Create vector with elements like 2010: +4.0%`
  year_percent <- paste(paste(years, ":", sep = ""), paste(pretty_percent, "%", sep = ""))
  
  # Collapse all years into single string
  paste(year_percent, collapse = ",")
  
  
  
  # A very simple table ----
  # Define the names vector
  income_names <- c("Year 0", "Year 1", "Year 2", "Project Lifetime")
  
  # Create pretty_income
  pretty_income <- format(income, digits = 2, big.mark = ",")
  
  # Create dollar_income
  dollar_income <- paste("$", pretty_income, sep = "")
  
  # Create formatted_names
  formatted_names <- format(income_names, justify = "right")
  
  # Create rows
  rows <- paste (formatted_names, dollar_income, sep = "   ")
  
  # Write rows
  writeLines(rows)
  
  
  
  # Let's order pizza! ----
  toppings <- c("anchovies",
                "artichoke",
                "bacon",
                "breakfast bacon",
                "Canadian bacon",
                "cheese",
                "chicken",
                "chili peppers",
                "feta",
                "garlic",
                "green peppers",
                "grilled onions",
                "ground beef",
                "ham",
                "hot sauce",
                "meatballs",
                "mushrooms",
                "olives",
                "onions",
                "pepperoni",
                "pineapple",
                "sausage",
                "spinach",
                "sun-dried tomato",
                "tomatoes")
  # Randomly sample 3 toppings
  my_toppings <- sample(toppings, size = 3)
  
  # Print my_toppings
  my_toppings
  
  # Paste "and " to last element: my_toppings_and
  my_toppings_and <- paste(c("","","and "), my_toppings, sep = "")
  
  # Collapse with comma space: these_toppings
  these_toppings <- paste(my_toppings_and, collapse = ", ")
  
  # Add rest of sentence: my_order
  my_order <- paste("I want to order a pizza with", paste(these_toppings, ".", sep = ""))
  
  # Order pizza with writeLines()
  writeLines(my_order)
  
  
  
# 2. Chapter 2: Introduction to stringr -----
  # Putting strings together with stringr ----
  library(stringr)
  
  my_toppings <- c("cheese", NA, NA)
  my_toppings_and <- paste(c("", "", "and "), my_toppings, sep = "")
  
  # Print my_toppings_and
  my_toppings_and
  
  # Use str_c() instead of paste(): my_toppings_str
  my_toppings_str <- str_c(c("", "", "and "), my_toppings)
  
  # Print my_toppings_str
  my_toppings_str
  
  # paste() my_toppings_and with collapse = ", "
  paste(my_toppings_and, collapse = ", ")
  
  # str_c() my_toppings_str with collapse = ", "
  str_c(my_toppings_str, collapse = ", ")
  
  
  
  # String length ----
  # Extracting vectors for boys' and girls' names
  babynames_2014 <- filter(babynames, year == 2014)
  boy_names <- filter(babynames_2014, sex == "M")$name
  girl_names <- filter(babynames_2014, sex == "F")$name
  
  # Take a look at a few boy_names
  head(boy_names)
  
  # Find the length of all boy_names
  boy_length <- str_length(boy_names)
  
  # Take a look at a few lengths
  head(boy_length)
  
  # Find the length of all girl_names
  girl_length <- str_length(girl_names)
  
  # Find the difference in mean length
  mean(girl_length) - mean(boy_length)
  
  # Confirm str_length() works with factors
  head(str_length(factor(boy_names)))
  
  # Just be aware this is a naive average where each name is counted once, 
  # not weighted by how many babies recevied the name. A better comparison might be an average weighted 
  # by the n column in babynames. 
  
  
  
  
  # Extracting substrings ----
  # The big advantage of str_sub() is the ability to use negative indexes to count from the end of a string. 
  # Extract first letter from boy_names
  boy_first_letter <- str_sub(boy_names, 1, 1)
  
  # Tabulate occurrences of boy_first_letter
  table(boy_first_letter)
  
  # Extract the last letter in boy_names, then tabulate
  boy_last_letter <- str_sub(boy_names, -1, -1)
  table(boy_last_letter)
  
  # Extract the first letter in girl_names, then tabulate
  girl_first_letter <- str_sub(girl_names,1, 1)
  table(girl_first_letter)
  
  # Extract the last letter in girl_names, then tabulate
  girl_last_letter <- str_sub(girl_names, -1, -1)
  table(girl_last_letter)
  
  
  
  # Detecting matches ----
  # Look for pattern "zz" in boy_names
  contains_zz <- str_detect(boy_names, "zz")
  
  # Examine str() of contains_zz
  str(contains_zz)
  
  # How many names contain "zz"?
  sum(contains_zz)
  
  # Which names contain "zz"?
  boy_names[contains_zz]
  

  
  
  # Subsetting strings based on match ----
  # Find boy_names that contain "zz"
  str_subset(boy_names, "zz")
  
  # Find girl_names that contain "zz"
  str_subset(girl_names, "zz")
  
  # Find girl_names that contain "U"
  starts_U <- str_subset(boy_names, "U")
  
  # Find girl_names that contain "U" and "z"
  str_subset(starts_U, "z")
  
  
  
  # Counting matches----
  # Count occurrences of "a" in girl_names
  number_as <- str_count(girl_names, "a")
  
  # Count occurrences of "A" in girl_names
  number_As <- str_count(girl_names, "A")
  
  # Histograms of number_as and number_As
  hist(number_as)
  hist(number_As)
  
  # Find total "a" + "A"
  total_as <- number_as + number_As
  
  # girl_names with more than 4 a's
  girl_names[total_as > 4]
  
  
  
  
  # Parsing strings into variables ----
  date_ranges <- c("23.01.2017 - 29.01.2017", "30.01.2017 - 06.02.2017")
  
  # Split dates using " - "
  split_dates <- str_split(date_ranges, fixed(" - "))
  
  # Print split_dates
  split_dates
  
  # Split dates with n and simplify specified
  split_dates_n <- str_split(date_ranges, fixed(" - "), simplify = TRUE, n = 2)
  split_dates_n
  
  # Subset split_dates_n into start_dates and end_dates
  start_dates <- split_dates_n[,1]
  end_dates <- split_dates_n[,2]
  
  # Split start_dates into day, month and year pieces
  str_split(start_dates, fixed("."), simplify =TRUE,  n = 3)
  
  # Split both_names into first_names and last_names
  both_names <- c("Box, George", "Cox, David")
  both_names_split <- str_split(both_names, fixed(", "), simplify = TRUE, n = 2)
  first_names <- both_names_split[,2]
  last_names <- both_names_split[,1]
  
  
  
  # Some simple text statistics ----
  lines <- c(
  "The table was a large one, but the three were all crowded together at one corner of it:"                                  
  , "\"No room! No room!\" they cried out when they saw Alice coming."                                                         
  , "\"There’s plenty of room!\" said Alice indignantly, and she sat down in a large arm-chair at one end of the table."
  )
  # Split lines into words
  words <- str_split(lines, fixed(" "))
  
  # Number of words per line
  lapply(words, length)
  
  # Number of characters in each word
  word_lengths <- lapply(words, str_length)
  
  # Average word length per line
  lapply(word_lengths, mean)
  
  
  
  # Replacing to tidy strings ----
  ids <- c("ID#: 192", "ID#: 118", "ID#: 001")
  
  # Replace "ID#: " with ""
  id_nums <- str_replace_all(ids, pattern = "ID#: ", replace = "")
  
  # Turn id_nums into numbers
  id_ints <- as.numeric(id_nums)
  
  # Some (fake) phone numbers
  phone_numbers <- c("510-555-0123", "541-555-0167")
  
  # Use str_replace() to replace "-" with " "
  str_replace(phone_numbers, pattern = "-", replace = " ")
  
  # Use str_replace_all() to replace "-" with " "
  str_replace_all(phone_numbers, pattern = "-", replace = " ")
  
  # Turn phone numbers into the format xxx.xxx.xxxx
  str_replace_all(phone_numbers, pattern = "-", replace = ".")
  
  
  
  
  # Review ----
  genes <- c(
  "TTAGAGTAAATTAATCCAATCTTTGACCCAAATCTCTGCTGGATCCTCTGGTATTTCATGTTGGATGACGTCAATTTCTAATATTTCACCCAACCGTTGAGCACCTTGTGCGATCAATTGTTGATCCAGTTTTATGATTGCACCGCAGAAAGTGTCATATTCTGAGCTGCCTAAACCAACCGCCCCAAAGCGTACTTGGGATAAATCAGGCTTTTGTTGTTCGATCTGTTCTAATAATGGCTGCAAGTTATCAGGTAGATCCCCGGCACCATGAGTGGATGTCACGATTAACCACAGGCCATTCAGCGTAAGTTCGTCCAACTCTGGGCCATGAAGTATTTCTGTAGAAAACCCAGCTTCTTCTAATTTATCCGCTAAATGTTCAGCAACATATTCAGCACTACCAAGCGTACTGCCACTTATCAACGTTATGTCAGCCAT" 
  , "TTAAGGAACGATCGTACGCATGATAGGGTTTTGCAGTGATATTAGTGTCTCGGTTGACTGGATCTCATCAATAGTCTGGATTTTGTTGATAAGTACCTGCTGCAATGCATCAATGGATTTACACATCACTTTAATAAATATGCTGTAGTGGCCAGTGGTGTAATAGGCCTCAACCACTTCTTCTAAGCTTTCCAATTTTTTCAAGGCGGAAGGGTAATCTTTGGCACTTTTCAAGATTATGCCAATAAAGCAGCAAACGTCGTAACCCAGTTGTTTTGGGTTAACGTGTACACAAGCTGCGGTAATGATCCCTGCTTGCCGCATCTTTTCTACTCTTACATGAATAGTTCCGGGGCTAACAGCGAGGTTTTTGGCTAATTCAGCATAGGGTGTGCGTGCATTTTCCATTAATGCTTTCAGGATGCTGCGATCGAGATTATCGATCTGATAAATTTCACTCAT" 
  , "ATGAAAAAACAATTTATCCAAAAACAACAACAAATCAGCTTCGTAAAATCATTCTTTTCCCGCCAATTAGAGCAACAACTTGGCTTGATCGAAGTCCAGGCTCCTATTTTGAGCCGTGTGGGTGATGGAACCCAAGATAACCTTTCTGGTTCTGAGAAAGCGGTACAGGTAAAAGTTAAGTCATTGCCGGATTCAACTTTTGAAGTTGTACATTCATTAGCGAAGTGGAAACGTAAAACCTTAGGGCGTTTTGATTTTGGTGCTGACCAAGGGGTGTATACCCATATGAAAGCATTGCGCCCAGATGAAGATCGCCTGAGTGCTATTCATTCTGTATATGTAGATCAGTGGGATTGGGAACGGGTTATGGGGGACGGTGAACGTAACCTGGCTTACCTGAAATCGACTGTTAACAAGATTTATGCAGCGATTAAAGAAACTGAAGCGGCGATCAGTGCTGAGTTTGGTGTGAAGCCTTTCCTGCCGGATCATATTCAGTTTATCCACAGTGAAAGCCTGCGGGCCAGATTCCCTGATTTAGATGCTAAAGGCCGTGAACGTGCAATTGCCAAAGAGTTAGGTGCTGTCTTCCTTATAGGGATTGGTGGCAAATTGGCAGATGGTCAATCCCATGATGTTCGTGCGCCAGATTATGATGATTGGACCTCTCCGAGTGCGGAAGGTTTCTCTGGATTAAACGGCGACATTATTGTCTGGAACCCAATATTGGAAGATGCCTTTGAGATATCTTCTATGGGAATTCGTGTTGATGCCGAAGCTCTTAAGCGTCAGTTAGCCCTGACTGGCGATGAAGACCGCTTGGAACTGGAATGGCATCAATCACTGTTGCGCGGTGAAATGCCACAAACTATCGGGGGAGGTATTGGTCAGTCCCGCTTAGTGATGTTATTGCTGCAGAAACAACATATTGGTCAGGTGCAATGTGGTGTTTGGGGCCCTGAAATCAGCGAGAAAGTTGATGGCCTGCTGTAA"
  )
  # Find the number of nucleotides in each sequence
  str_length(genes)
  
  # Find the number of A's occur in each sequence
  str_count(genes, pattern = "A")
  
  # Return the sequences that contain "TTTTTT"
  str_subset(genes, pattern = "TTTTTT")
  
  # Replace all the "A"s in the sequences with a "_"
  str_replace_all(genes, pattern = "A", replace = "_")
  
  
  
  # Final challenges ----
  # --- Task 1 ----
  # Define some full names
  names <- c("Diana Prince", "Clark Kent")
  
  # Split into first and last names
  names_split <- str_split(names, fixed(" "), simplify = TRUE, n = 2)
  
  # Extract the first letter in the first name
  abb_first <- str_sub(names_split[,1], 1, 1)
  
  # Combine the first letter ". " and last name
  str_c(abb_first, c(". ",". "), names_split[,2])
  
  # --- Task 2 ----
  # Use all names in babynames_2014
  all_names <- babynames_2014$name
  
  # Get the last two letters of all_names
  last_two_letters <- str_sub(all_names, -2, -1)
  
  # Does the name end in "ee"?
  ends_in_ee <- str_detect(last_two_letters, "ee")
  
  # Extract rows and "sex" column
  sex <- babynames_2014[ends_in_ee,]$sex
  
  # Display result as a table
  table(sex)
  
  
  
  
# 3. Chapter 3: Pattern matching with regular expressions -----
  # Matching the start or end of the string ----
  library(rebus)
  library(stringr)
  
  # Some strings to practice with
  x <- c("cat", "coat", "scotland", "tic toc")
  
  # Print END
  END
  
  # Run me
  str_view(x, pattern = START %R% "c")
  
  # Match the strings that start with "co" 
  str_view(x, pattern = START %R% "co")
  
  # Match the strings that end with "at"
  str_view(x, pattern = "at" %R% END)
  
  # Match the strings that is exactly "cat"
  str_view(x, pattern = START %R% "cat" %R% END)
  
  
  
  # Matching any character ----
  x <- c("cat", "coat", "scotland", "tic toc")
  
  # Match any character followed by a "t"
  str_view(x, pattern = ANY_CHAR %R% "t")
  
  # Match a "t" followed by any character
  str_view(x, pattern = "t" %R% ANY_CHAR)
  
  # Match two characters
  str_view(x, pattern = ANY_CHAR %R% ANY_CHAR)
  
  # Match a string with exactly three characters
  str_view(x, pattern = START %R% ANY_CHAR %R% ANY_CHAR %R% ANY_CHAR %R% END)
  
  
  
  # Combining with stringr functions ----
  # q followed by any character
  pattern <- "q" %R% ANY_CHAR
  
  # Test pattern 
  str_view(c("Quentin", "Kaliq", "Jacques",  "Jacqes"), pattern)  
  
  # Find names that have the pattern
  names_with_q <- str_subset(boy_names, pattern)
  length(names_with_q)
  
  # Find part of name that matches pattern
  part_with_q <- str_extract(boy_names, pattern)
  table(part_with_q)
  
  # Did any names have the pattern more than once?
  count_of_q <- str_count(boy_names, pattern)
  table(count_of_q)
  
  # How many babies got these names?
  with_q <- str_detect(boy_names, pattern)
  boy_df[with_q, ]
  
  
  
  
  # Alternation ----
  # Match Jeffrey or Geoffrey
  whole_names <- or("Jeffrey", "Geoffrey")
  str_view(boy_names, pattern = whole_names, 
           match = TRUE)
  
  # Match Jeffrey or Geoffrey, another way
  common_ending <- or("Je", "Geo") %R% "ffrey"
  str_view(boy_names, pattern = common_ending, 
           match = TRUE)
  
  # Match with alternate endings
  by_parts <- or("Je", "Geo") %R% "ff" %R% or("ry", "ery", "rey", "erey")
  str_view(boy_names, 
           pattern = by_parts, 
           match = TRUE)
  
  # Match names that start with Cath or Kath
  ckath <- START %R% or("C","K") %R% "ath"
  str_view(girl_names, pattern = ckath, match = TRUE)
  
  
  
  # Character classes ----
  x <- c( "grey sky", "gray elephant")
  # Create character class containing vowels
  vowels <- char_class("aeiouAEIOU")
  
  # Print vowels
  vowels
  
  # See vowels in x with str_view()
  str_view(x, pattern = vowels, match = TRUE)
  
  # See vowels in x with str_view_all()
  str_view_all(x, pattern = vowels, match = TRUE)
  
  # Number of vowels in boy_names
  num_vowels <- str_count(boy_names, pattern = vowels)
  mean(num_vowels)
  
  # Proportion of vowels in boy_names
  name_length <- str_length(boy_names)
  mean(num_vowels/name_length)
  
  
  # Repetition ----
  # Vowels from last exercise
  vowels <- char_class("AEIOUaeiou")
  
  # Use `negated_char_class()` for everything but vowels
  not_vowels <- negated_char_class("AEIOUaeiou")
  
  # See names with only vowels
  str_view(boy_names, 
           pattern = START %R% one_or_more(vowels) %R% END, 
           match = TRUE)
  
  # See names with no vowels
  str_view(boy_names, 
           pattern = START %R% one_or_more(not_vowels) %R% END, 
           match = TRUE)
  
  
  
  # Hunting for phone numbers ----
  contact <- c(
    "Call me at 555-555-0191",                 
    "123 Main St",                             
    "(555) 555 0191",                          
    "Phone: 555.555.0191 Mobile: 555.555.0192"
    )
  
  # Take a look at ALL digits
  str_view_all(contact, DGT)
  
  # Create a three digit pattern and test
  three_digits <- DGT %R% DGT %R% DGT
  str_view_all(contact,
               pattern = three_digits)
  
  # Create four digit pattern
  four_digits <- three_digits %R% DGT
  
  # Create a separator pattern and test
  separator <- char_class("-.() ")
  str_view_all(contact,
               pattern = separator)
  
  # Create phone pattern
  phone_pattern <- three_digits %R% 
    zero_or_more(separator) %R% 
    three_digits %R% 
    zero_or_more(separator) %R%
    four_digits
  
  # Test pattern           
  str_view(contact, pattern = phone_pattern)
  
  # Extract phone numbers
  str_extract(contact, phone_pattern)
  
  # Extract ALL phone numbers
  str_extract_all(contact, phone_pattern)
  
  
  
  # Extracting age and gender from accident narratives ----
  narratives <- c(
    "19YOM-SHOULDER STRAIN-WAS TACKLED WHILE PLAYING FOOTBALL W/ FRIENDS ",                      
    "31 YOF FELL FROM TOILET HITITNG HEAD SUSTAINING A CHI ",                                    
    "ANKLE STR. 82 YOM STRAINED ANKLE GETTING OUT OF BED ",                                      
    "TRIPPED OVER CAT AND LANDED ON HARDWOOD FLOOR. LACERATION ELBOW, LEFT. 33 YOF*",            
    "10YOM CUT THUMB ON METAL TRASH CAN DX AVULSION OF SKIN OF THUMB ",                          
    "53 YO F TRIPPED ON CARPET AT HOME. DX HIP CONTUSION ",                                      
    "13 MOF TRYING TO STAND UP HOLDING ONTO BED FELL AND HIT FOREHEAD ON RADIATOR DX LACERATION",
    "14YR M PLAYING FOOTBALL; DX KNEE SPRAIN ",                                                  
    "55YOM RIDER OF A BICYCLE AND FELL OFF SUSTAINED A CONTUSION TO KNEE ",                      
    "5 YOM ROLLING ON FLOOR DOING A SOMERSAULT AND SUSTAINED A CERVICAL STRA IN"
    )
  # Look for two digits
  str_view(narratives, pattern = DGT %R% DGT)
  
  # Pattern to match one or two digits
  age <- or(DGT, DGT %R% DGT)
  str_view(narratives, 
           pattern = age)
  
  # Pattern to match units 
  unit <- zero_or_more(" ") %R% or("YO", "YR", "MO")
  
  # Test pattern with age then units
  str_view(narratives, 
           pattern = age %R% unit)
  
  # Pattern to match gender
  gender <- zero_or_more(" ") %R% or("M", "F")
  
  # Test pattern with age then units then gender
  str_view(narratives, 
           pattern = age %R% unit %R% gender)
  
  # Extract age_gender, take a look
  age_gender <- str_extract(narratives, age %R% unit %R% gender)
  age_gender
  
  
  
  # Parsing age and gender into pieces ----
  # Extract age and make numeric
  ages_numeric <- as.numeric(str_extract(age_gender, pattern = age))
  
  # Replace age and units with ""
  genders <- str_replace(age_gender, 
                         pattern = age %R% unit, 
                         replacement = "")
  
  # Replace extra spaces
  genders_clean <- str_replace_all(genders, 
                                   pattern = one_or_more(" "), 
                                   replacement = "")
  
  # Extract units 
  time_units <- str_extract(age_gender, pattern = unit)
  
  # Extract first word character
  time_units_clean <- str_extract(time_units, pattern = WRD)
  
  # Turn ages in months to years
  ages_years <- ifelse(time_units_clean == "Y", ages_numeric, ages_numeric/12)

  
  
  
  
# 4. Chapter 4: More advanced matching and manipulation  -----
  # !!! Capturing parts of a pattern ----
  hero_contacts <- c(
    "(wolverine@xmen.com)",
    "wonderwoman@justiceleague.org",
    "thor@avengers.com"
    )
  # Capture part between @ and . and after .
  email <- capture(one_or_more(WRD)) %R% 
    "@" %R% capture(one_or_more(WRD)) %R% 
    DOT %R% capture(one_or_more(WRD))
  
  # Check match hasn't changed
  str_view(hero_contacts, pattern = email)
  
  # Pull out match and captures
  email_parts <- str_match(hero_contacts, pattern = email)
  
  # Print email_parts
  email_parts
  
  # Save host
  host <- email_parts[,3]
  host
  
  # detecting an email address can be really hard see this discussion for more details.
  # http://stackoverflow.com/questions/201323/using-a-regular-expression-to-validate-an-email-address/201378#201378
  
  
  
  
  
  # Pulling out parts of a phone number  (s.o.)----
  # View text containing phone numbers
  contact
  
  # Add capture() to get digit parts
  phone_pattern <- capture(three_digits) %R% zero_or_more(separator) %R% 
    capture(three_digits) %R% zero_or_more(separator) %R%
    capture(four_digits)
  
  # Pull out the parts with str_match()
  phone_numbers <- str_match(contact, pattern = phone_pattern)
  
  # Put them back together
  str_c(
    "(",
    phone_numbers[,2],
    ")",
    phone_numbers[,3],
    "-",
    phone_numbers[,4]
  )
  
  # The second phone number in the last string, you could use str_match_all(). 
  # But, like str_split() it will return a list with one component for each input string, 
  # and you'll need to use lapply() to handle the result. 
  
  
  
  # Extracting age and gender again -----
  # narratives has been pre-defined
  narratives
  
  # Add capture() to get age, unit and sex
  pattern <- capture(optional(DGT) %R% DGT) %R%  
    optional(SPC) %R% capture(or("YO", "YR", "MO")) %R%
    optional(SPC) %R% capture(or("M", "F"))
  
  # Pull out from narratives
  str_match(narratives, pattern = pattern)
  
  # Edit to capture just Y and M in units
  pattern2 <- capture(optional(DGT) %R% DGT) %R%  
    optional(SPC) %R% capture(or("Y", "M")) %R% optional(or("O","R")) %R%
    optional(SPC) %R% capture(or("M", "F"))
  
  # Check pattern
  str_view(narratives, pattern2)
  
  # Pull out pieces
  str_match(narratives, pattern2)
  
  
  
  
  # Using backreferences in patterns ----
  boy_names <- tolower((boy_names))
  # See names with three repeated letters
  repeated_three_times <- capture(LOWER) %R% REF1 %R% REF1
  str_view(boy_names, 
           pattern = repeated_three_times, 
           match = TRUE)
  
  # See names with a pair of repeated letters
  pair_of_repeated <- capture(LOWER %R% LOWER) %R% REF1
  str_view(boy_names, 
           pattern = pair_of_repeated, 
           match = TRUE)
  
  # See names with a pair that reverses
  pair_that_reverses <- capture(LOWER) %R% capture(LOWER) %R% REF2 %R% REF1
  str_view(boy_names, 
           pattern = pair_that_reverses, 
           match = TRUE)
  
  # See four letter palindrome names
  four_letter_palindrome <- exactly(capture(LOWER) %R% capture(LOWER) %R% REF2 %R% REF1)
  str_view(boy_names, 
           pattern = four_letter_palindrome, 
           match = TRUE)
  
  # See six letter palindrome names
  six_letter_palindrome <- exactly(capture(LOWER) %R% capture(LOWER) %R% capture(LOWER) %R% REF3 %R% REF2 %R% REF1)
  str_view(boy_names, 
           pattern = six_letter_palindrome, 
           match = TRUE)
  
  
  
  
  # Replacing with regular expressions ----
  # View text containing phone numbers
  contact
  
  # Replace digits with "X"
  str_replace(contact, pattern = DGT, replacement = "X")
  
  # Replace all digits with "X"
  str_replace_all(contact, pattern = DGT, replacement = "X")
  
  # !!!! Replace all digits with different symbol
  str_replace_all(contact, pattern = DGT, 
                  replacement = c("X", ".", "*", "_"))
  
  
  
  # Replacing with backreferences ----
  # Build pattern to match words ending in "ING"
  pattern <- one_or_more(WRD) %R% "ING"
  str_view(narratives, pattern)
  
  # Test replacement
  str_replace(narratives, capture(pattern), str_c("CARELESSLY", REF1, sep = " "))
  
  # One adverb per narrative
  adverbs_10 <- sample(adverbs, 10)
  
  # Replace "***ing" with "adverb ***ing"
  str_replace(narratives, capture(pattern), str_c(adverbs_10, REF1, sep = " "))
  
  
  
  # Matching a specific code point or code groups ----
  # Names with builtin accents
  (tay_son_builtin <- c(
    "Nguy\u1ec5n Nh\u1ea1c", 
    "Nguy\u1ec5n Hu\u1ec7",
    "Nguy\u1ec5n Quang To\u1ea3n"
  ))
  
  # Convert to separate accents
  tay_son_separate <- stri_trans_nfd(tay_son_builtin)
  
  # Verify that the string prints the same
  tay_son_separate
  
  # Match all accents
  str_view_all(tay_son_separate, UP_DIACRITIC)
  
  
  
  # Matching a single grapheme ----
  # tay_son_separate has been pre-defined
  tay_son_separate
  
  # View all the characters in tay_son_separate
  str_view_all(tay_son_separate, ANY_CHAR)
  
  # View all the graphemes in tay_son_separate
  str_view_all(tay_son_separate, GRAPHEME)
  
  # Combine the diacritics with their letters
  tay_son_builtin <- stri_trans_nfc(tay_son_separate)
  
  # View all the graphemes in tay_son_builtin
  str_view_all(tay_son_builtin, GRAPHEME)
  
  
  
  
# 5. Chapter 5: Case studies -----
  # Getting the play into R ----
  url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_2922/datasets/importance-of-being-earnest.txt"
  download.file(url, "importance-of-being-earnest.txt")
  earnest <- stri_read_lines("importance-of-being-earnest.txt")
  
  # Detect start and end lines
  start <- which(str_detect(earnest, pattern = "START OF THE PROJECT"))
  end   <- which(str_detect(earnest, pattern = "END OF THE PROJECT"))
  
  # Get rid of gutenberg intro text
  earnest_sub  <- earnest[(start + 1):(end - 1)]
  
  # Detect first act
  lines_start <- which(str_detect(earnest_sub, pattern = "FIRST ACT"))
  
  # Set up index
  intro_line_index <- 1:(lines_start - 1)
  
  # Split play into intro and play
  intro_text <- earnest_sub[intro_line_index]
  play_text <-  earnest_sub[-intro_line_index]
  
  # Take a look at the first 20 lines
  writeLines(play_text[1:20])
  
  
  
  # Identifying the lines, take 1 ----
  
  # # Get rid of empty strings
  # empty <- stri_isempty(play_text)
  # play_lines <- play_text[!empty]
  
  # Pattern for start word then .
  pattern_1 <- START %R% one_or_more(WRD) %R% DOT
  
  # Test pattern_1
  str_view(play_lines, pattern = pattern_1, 
           match = TRUE) 
  str_view(play_lines, pattern = pattern_1, 
           match = FALSE)
  
  # Pattern for start, capital, word then .
  pattern_2 <- START %R% ascii_upper() %R% one_or_more(WRD) %R% DOT
  
  # View matches of pattern_2
  str_view(play_lines, pattern = pattern_2, 
           match = TRUE) 
  
  # View non-matches of pattern_2
  str_view(play_lines, pattern = pattern_2, 
           match = FALSE) 
  
  # Get subset of lines that match
  lines <- str_subset(play_lines, pattern_2)
  
  # Extract match from lines
  who <- str_extract(lines, pattern_2)
  
  # Let's see what we have
  unique(who)
  
  # Good job, but it looks like your pattern wasn't 100% successful. 
  # It missed Lady Bracknell, and picked up lines starting with University., July. and a few others. 
  # Let's try a slighty different strategy. 
  
  
  
  # Identifying the lines, take 2 ----
  # Create vector of characters
  characters <- c("Algernon", "Jack", "Lane", "Cecily", "Gwendolen", "Chasuble", 
                  "Merriman", "Lady Bracknell", "Miss Prism")
  
  # !!!! Match start, then character name, then .
  pattern_3 <- START %R% or1(characters) %R% DOT
  
  # View matches of pattern_3
  str_view(play_lines, pattern_3, match = TRUE)
  
  # View non-matches of pattern_3
  str_view(play_lines, pattern_3, match = FALSE)
  
  # Pull out matches
  lines <- str_subset(play_lines, pattern_3)
  
  # Extract match from lines
  who <- str_extract(lines, pattern_3)
  
  # Let's see what we have
  unique(who)
  
  # Count lines per character
  table(who)
  
  # Changing case to ease matching ----
  
  catcidents <- c(
    "79yOf Fractured fingeR tRiPPED ovER cAT ANd fell to FlOOr lAst nIGHT AT HOME*"                                                                  
    ,"21 YOF REPORTS SUS LACERATION OF HER LEFT HAND WHEN SHE WAS OPENING A CAN OF CAT FOOD JUST PTA. DX HAND LACERATION%"                            
    ,"87YOF TRIPPED OVER CAT, HIT LEG ON STEP. DX LOWER LEG CONTUSION "                                                                               
    ,"bLUNT CHest trAUma, R/o RIb fX, R/O CartiLAgE InJ To RIB cAge; 32YOM walKiNG DOG, dog took OfF aFtER cAt,FelL,stRucK CHest oN STepS,hiT rIbS"   
    ,"42YOF TO ER FOR BACK PAIN AFTER PUTTING DOWN SOME CAT LITTER DX: BACK PAIN, SCIATICA"                                                           
    ,"4YOf DOg jUst hAd PUpPieS, Cat TRIED 2 get PuPpIes, pT THru CaT dwn stA Irs, LoST foOTING & FELl down ~12 stePS; MInor hEaD iNJuRY"             
    ,"unhelmeted 14yof riding her bike with her dog when she saw a cat and sw erved c/o head/shoulder/elbow pain.dx: minor head injury,left shoulder" 
    ,"24Yof lifting a 40 pound bag of cat litter injured lt wrist; wrist sprain"                                                                      
    ,"3Yof-foot lac-cut on cat food can-@ home "                                                                                                      
    ,"Rt Shoulder Strain.26Yof Was Walking Dog On Leash And Dot Saw A Cat And Pulled Leash."                                                          
    ,"15 mO m cut FinGer ON cAT FoOd CAn LID. Dx:  r INDeX laC 1 cm."                                                                                 
    ,"31 YOM SUSTAINED A CONTUSION OF A HAND BY TRIPPING ON CAT & FALLING ON STAIRS."                                                                 
    ,"ACCIDENTALLY CUT FINGER WHILE OPENING A CAT FOOD CAN, +BLEEDING >>LAC"                                                                          
    ,"4 Yom was cut on cat food can. Dx:  r index lac 1 cm."                                                                                          
    ,"4 YO F, C/O FOREIGN BODY IN NOSE 1/2 HOUR, PT NOT REPORTING NATURE OF F B, PIECE OF CAT LITTER REMOVED FROM RT NOSTRIL, DX FB NOSE"             
    ,"21Yowf  pT STAteS 4-5 DaYs Ago LifTEd 2 - 50 lB BagS OF CAT lItter.  al So sORTIng ClOThES & W/ seVERe paIn.  DX .  sTrain  LUMbOSaCRal."       
    ,"67 YO F WENT TO WALK DOG, IT STARTED TO CHASE CAT JERKED LEASH PULLED H ER OFF PATIO, FELL HURT ANKLES. DX BILATERAL ANKLE FRACTURES"           
    ,"17Yof Cut Right Hand On A Cat Food Can - Laceration "                                                                                           
    ,"46yof taking dog outside, dog bent her fingers back on a door. dog jerk ed when saw cat. hand holding leash caught on door jamb/ct hand"        
    ,"19 YOF-FelL whIle WALKINg DOWn THE sTAIrS & TRiPpEd over a caT-fell oNT o \"TaIlBoNe\"         dx   coNtusIon LUMBaR, uti      *"               
    ,"50YOF CUT FINGER ON CAT FOOD CAN LID.  DX: LT RING FINGER LAC "                                                                                 
    ,"lEFT KNEE cOntusioN.78YOf triPPEd OVEr CaT aND fell and hIt knEE ON the fLoOr."                                                                 
    ,"LaC FInGer oN a meTAL Cat fOOd CaN "                                                                                                            
    ,"PUSHING HER UTD WITH SHOTS DOG AWAY FROM THE CAT'S BOWL&BITTEN TO FINGE R>>PW/DOG BITE"                                                         
    ,"DX CALF STRAIN R CALF: 15YOF R CALF PN AFTER FALL ON CARPETED STEPS, TR YING TO STEP OVER CAT, TRIPPED ON STAIRS, HIT LEG"                      
    ,"DISLOCATION TOE - 80 YO FEMALE REPORTS SHE FELL AT HOME - TRIPPED OVER THE CAT LITTER BOX & FELL STRIKING TOE ON DOOR JAMB - ALSO SHOULDER INJ" 
    ,"73YOF-RADIUS FX-TRIPPED OVER CAT LITTER BOX-FELL-@ HOME "                                                                                       
    ,"57Yom-Back Pain-Tripped Over A Cat-Fell Down 4 Steps-@ Home "                                                                                   
    ,"76YOF SUSTAINED A HAND ABRASION CLEANING OUT CAT LITTER BOX THREE DAYS AGO AND NOW THE ABRASION IS INFECTED CELLULITIS HAND"                    
    ,"DX R SH PN: 27YOF W/ R SH PN X 5D. STATES WAS YANK' BY HER DOG ON LEASH W DOG RAN AFTER CAT; WORSE' PN SINCE. FULL ROM BUT VERY PAINFUL TO MOVE"
    ,"35Yof FeLt POp iN aBdoMeN whIlE piCKInG UP 40Lb BaG OF CAt litTeR aBdomINAL sTrain"                                                             
    ,"77 Y/o f tripped over cat-c/o shoulder and upper arm pain. Fell to floo r at home. Dx proximal humerus fx"                                      
    ,"FOREHEAD LAC.46YOM TRIPPED OVER CAT AND FELL INTO A DOOR FRAME. "                                                                               
    ,"39Yof dog pulled her down the stairs while chasing a cat dx: rt ankle inj"                                                                      
    ,"10 YO FEMALE OPENING A CAN OF CAT FOOD.  DX HAND LACERATION "                                                                                   
    ,"44Yof Walking Dog And The Dof Took Off After A Cat And Pulled Pt Down B Y The Leash Strained Neck"                                              
    ,"46Yof has low back pain after lifting heavy bag of cat litter lumbar spine sprain"                                                              
    ,"62 yOf FELL PUShIng carT W/CAT liTtER 3 DAYs Ago. Dx:  l FIfTH rib conT."                                                                       
    ,"PT OPENING HER REFRIGERATOR AND TRIPPED OVER A CAT AND FELL ONTO SHOULD ER FRACTURED HUMERUS"                                                   
    ,"Pt Lifted Bag Of Cat Food. Dx:  Low Back Px, Hx Arthritic Spine."
  )
  
  # catcidents has been pre-defined
  head(catcidents)
  
  # Construct pattern of DOG in boundaries
  whole_dog_pattern <- whole_word("DOG")
  
  # View matches to word "DOG"
  str_view(catcidents, whole_dog_pattern, match = TRUE)
  
  # Transform catcidents to upper case
  catcidents_upper <- str_to_upper(catcidents)
  
  # View matches to word "DOG" again
  str_view(catcidents_upper, whole_dog_pattern, match = TRUE)
  
  # Which strings match?
  has_dog <- str_detect(catcidents_upper, whole_dog_pattern)
  
  # Pull out matching strings in original 
  catcidents[has_dog]
  
  
  
  
  # Ignoring case when matching ----
  # View matches to "TRIP"
  str_view(catcidents, "TRIP", match = TRUE)
  
  # Construct case insensitive pattern
  trip_pattern <- regex("TRIP", ignore_case = TRUE)
  
  # View case insensitive matches to "TRIP"
  str_view(catcidents, trip_pattern, match = TRUE)
  
  # Get subset of matches
  trip <- str_subset(catcidents, trip_pattern)
  
  # Extract matches
  str_extract(trip, trip_pattern)
  
  
  
  # Fixing case problems ----
  library(stringi)
  
  # Get first five catcidents
  cat5 <- catcidents[1:5]
  
  # Take a look at original
  writeLines(cat5)
  
  # Transform to title case
  writeLines(str_to_title(cat5))
  
  # Transform to title case with stringi
  writeLines(stri_trans_totitle(cat5))
  
  # Transform to sentence case with stringi
  writeLines(stri_trans_totitle(cat5, type = "sentence"))