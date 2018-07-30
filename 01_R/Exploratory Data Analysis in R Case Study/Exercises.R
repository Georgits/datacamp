descriptions <- readRDS("descriptions.rds")
votes <- readRDS("votes.rds")


# Chapter 1: Data cleaning and summarizing with dplyr ----
# 1.1. Filtering rows ----
    # Load the dplyr package
    library(dplyr)
    
    # Print the votes dataset
    print(votes)
    
    # Filter for votes that are "yes", "abstain", or "no"
    votes %>%
      filter(vote <= 3)
    
# 1.2. Adding a year column ----
    # Add another %>% step to add a year column
    votes %>%
      filter(vote <= 3) %>%
      mutate(year = session + 1945)
    
# 1.3. Adding a country column ----
    # Load the countrycode package
    library(countrycode)
    
    # Convert country code 100
    countrycode(100, "cown", "country.name")
    
    # Add a country column within the mutate: votes_processed
    votes_processed <- votes %>%
      filter(vote <= 3) %>%
      mutate(year = session + 1945, country = countrycode(ccode, "cown", "country.name"))
    
# 1.4. Summarizing the full dataset ----
    # Print votes_processed
    print(votes_processed)
    
    # Find total and fraction of "yes" votes
    votes_processed %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
# 1.5. Summarizing by year ----
    # Change this code to summarize by year
    votes_processed %>%
      group_by(year) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
# 1.6. Summarizing by country ----
    # Summarize by country: by_country
    by_country <- votes_processed %>%
      group_by(country) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
# 1.7. Sorting by percentage of "yes" votes ----
    # You have the votes summarized by country
    by_country <- votes_processed %>%
      group_by(country) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
    # Print the by_country dataset
    print(by_country)
    
    # Sort in ascending order of percent_yes
    by_country %>%
      arrange(percent_yes)
    
    # Now sort in descending order
    by_country %>%
      arrange(desc(percent_yes))
    
# 1.8. Filtering summarized output ----
    # Filter out countries with fewer than 100 votes
    by_country %>%
      filter(total >= 100) %>%
      arrange(percent_yes)
    
    
# Chapter 2: Data visualization with ggplot2 ----
# 2.1. Plotting a line over time ----
    # Define by_year
    by_year <- votes_processed %>%
      group_by(year) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
    # Load the ggplot2 package
    library(ggplot2)
    
    # Create line plot
    ggplot(by_year, aes(x = year, y = percent_yes)) + 
      geom_line()
    
# 2.2. Other ggplot2 layers ----
    # Change to scatter plot and add smoothing curve
    ggplot(by_year, aes(year, percent_yes)) +
      geom_point() + 
      geom_smooth()

# 2.3. Summarizing by year and country ----
    # Group by year and country: by_year_country
    by_year_country <- votes_processed %>%
      group_by(year, country) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
# 2.4. Plotting just the UK over time ----
    # Start with by_year_country dataset
    by_year_country <- votes_processed %>%
      group_by(year, country) %>%
      summarize(total = n(),
                percent_yes = mean(vote == 1))
    
    # Print by_year_country
    print(by_year_country)
    
    # Create a filtered version: UK_by_year
    UK_by_year <- by_year_country %>%
      filter(country == "United Kingdom")
    
    # Line plot of percent_yes over time for UK only
    ggplot(UK_by_year, aes(x = year, y = percent_yes)) +
      geom_line()
    
# 2.5. Plotting multiple countries ----
    # Vector of four countries to examine
    countries <- c("United States", "United Kingdom",
                   "France", "India")
    
    # Filter by_year_country: filtered_4_countries
    filtered_4_countries <- by_year_country %>%
      filter(country %in% countries)
    
    # Line plot of % yes in four countries
    ggplot(filtered_4_countries, aes(x = year, y = percent_yes, color = country)) +
      geom_line()
    
# 2.6. Faceting by country ----
    # Vector of six countries to examine
    countries <- c("United States", "United Kingdom",
                   "France", "Japan", "Brazil", "India")
    
    # Filtered by_year_country: filtered_6_countries
    filtered_6_countries <- by_year_country %>%
      filter(country %in% countries)
    
    # Line plot of % yes over time faceted by country
    ggplot(filtered_6_countries, aes(x = year, y = percent_yes, color = country)) +
      geom_line() + 
      facet_wrap(~ country)
    
# 2.7. Faceting with free y-axis ----
    # Vector of six countries to examine
    countries <- c("United States", "United Kingdom",
                   "France", "Japan", "Brazil", "India")
    
    # Filtered by_year_country: filtered_6_countries
    filtered_6_countries <- by_year_country %>%
      filter(country %in% countries)
    
    # Line plot of % yes over time faceted by country
    ggplot(filtered_6_countries, aes(year, percent_yes)) +
      geom_line() +
      facet_wrap(~ country, scale = "free_y")
    
# 2.8. Choose your own countries ----
    # Add three more countries to this list
    countries <- c("United States", "United Kingdom",
                   "France", "Japan", "Brazil", "India", "Germany", "Russian Federation", "Georgia")
    
    # Filtered by_year_country: filtered_countries
    filtered_countries <- by_year_country %>%
      filter(country %in% countries)
    
    # Line plot of % yes over time faceted by country
    ggplot(filtered_countries, aes(year, percent_yes)) +
      geom_line() +
      facet_wrap(~ country, scales = "free_y")
    
    
# Chapter 3: Tidy modeling with broom ----
# 3.1. Linear regression on the United States ----
    # Percentage of yes votes from the US by year: US_by_year
    US_by_year <- by_year_country %>%
      filter(country == "United States")
    
    # Print the US_by_year data
    print(US_by_year)
    
    # Perform a linear regression of percent_yes by year: US_fit
    US_fit <- lm(percent_yes ~ year, data = US_by_year)
    
    # Perform summary() on the US_fit object
    summary(US_fit)
    
# 3.2. Tidying a linear regression model ----
    # Load the broom package
    library(broom)
    
    # Call the tidy() function on the US_fit object
    tidy(US_fit)
    
# 3.3. Combining models for multiple countries ----
    # Linear regression of percent_yes by year for US
    US_by_year <- by_year_country %>%
      filter(country == "United States")
    US_fit <- lm(percent_yes ~ year, US_by_year)
    
    # Fit model for the United Kingdom
    UK_by_year <- by_year_country %>%
      filter(country == "United Kingdom")
    UK_fit <- lm(percent_yes ~ year, UK_by_year)
    
    # Create US_tidied and UK_tidied
    US_tidied <- tidy(US_fit)
    UK_tidied <- tidy(UK_fit)
    
    # Combine the two tidied models
    US_tidied %>%
      bind_rows(UK_tidied)
    
# 3.4. Nesting a data frame ----
    # Load the tidyr package
    library(tidyr)
    
    # Nest all columns besides country
    by_year_country %>%
      nest( -country)
    
    
# 3.5. Nesting a data frame ----
    # All countries are nested besides country
    nested <- by_year_country %>%
      nest(-country)
    
    # Print the nested data for Brazil
    nested$data[[7]]
    
# 3.6. Unnesting ----
    # All countries are nested besides country
    nested <- by_year_country %>%
      nest(-country)
    
    # Unnest the data column to return it to its original form
    nested %>%
      unnest()
    
# 3.6. Performing linear regression on each nested dataset ----
    # Load tidyr and purrr
    library(tidyr)
    library(purrr)
    
    # Perform a linear regression on each item in the data column
    by_year_country %>%
      nest(-country) %>%
      mutate(model = map(data, ~ lm(percent_yes ~ year, data = .)))
    
    
# 3.7. Tidy each linear regression model   ----
    # Load the broom package
    library(broom)
    
    # Add another mutate that applies tidy() to each model
    by_year_country %>%
      nest(-country) %>%
      mutate(model = map(data, ~ lm(percent_yes ~ year, data = .))) %>%
      mutate(tidied = map(model, tidy))
    
# 3.8. Unnesting a data frame   ----
    # Add one more step that unnests the tidied column
    country_coefficients <- by_year_country %>%
      nest(-country) %>%
      mutate(model = map(data, ~ lm(percent_yes ~ year, data = .)),
             tidied = map(model, tidy)) %>%
      unnest(tidied)
    
    # Print the resulting country_coefficients variable
    print(country_coefficients)
    
# 3.9. Filtering model terms   ----
    # Print the country_coefficients dataset
    print(country_coefficients)
    
    # Filter for only the slope terms
    country_coefficients %>%
      filter(term == "year")
    
# 3.10. Filtering for significant countries   ----
    # Filter for only the slope terms
    slope_terms <- country_coefficients %>%
      filter(term == "year")
    
    # Add p.adjusted column, then filter
    slope_terms %>%
      mutate(p.adjusted = p.adjust(p.value))
    
# 3.11. Sorting by slope   ----
    # Filter by adjusted p-values
    filtered_countries <- country_coefficients %>%
      filter(term == "year") %>%
      mutate(p.adjusted = p.adjust(p.value)) %>%
      filter(p.adjusted < .05)
    
    # Sort for the countries increasing most quickly
    filtered_countries %>%
      arrange(desc(estimate))
    
    # Sort for the countries decreasing most quickly
    filtered_countries %>%
      arrange(estimate)
    
    
# Chapter 4: Joining and tidying ----
# 4.1. Joining datasets with inner_join ----
    # Load dplyr package
    library(dplyr)
    
    # Print the votes_processed dataset
    print(votes_processed)
    
    # Print the descriptions dataset
    print(descriptions)
    
    # Join them together based on the "rcid" and "session" columns
    votes_joined <- votes_processed %>%
      inner_join(descriptions, by = c("rcid", "session"))
    
# 4.2. Joining datasets with inner_join ----
    # Filter for votes related to colonialism
    votes_joined %>%
      filter(co == 1)
    
# 4.3. Visualizing colonialism votes ----
    # Load the ggplot2 package
    library(ggplot2)
    
    # Filter, then summarize by year: US_co_by_year
    US_co_by_year <- votes_joined %>%
      filter(co == 1 & country == "United States") %>%
      group_by(year) %>%
      summarize(percent_yes = mean(vote == 1))
    
    # Graph the % of "yes" votes over time
    ggplot(US_co_by_year, aes(x = year, y = percent_yes)) +
      geom_line()
    
# 4.4. Using gather to tidy a dataset ----
    # Load the tidyr package
    library(tidyr)
    
    # Gather the six mu/nu/di/hr/co/ec columns
    votes_joined %>%
      gather(topic, has_topic, me:ec)
    
    # Perform gather again, then filter
    votes_gathered <- votes_joined %>%
      gather(topic, has_topic, me:ec) %>%
      filter(has_topic == 1)
    
# 4.5. Using gather to tidy a dataset ----
    # Replace the two-letter codes in topic: votes_tidied
    votes_tidied <- votes_gathered %>%
      mutate(topic = recode(topic,
                            me = "Palestinian conflict",
                            nu = "Nuclear weapons and nuclear material",
                            di = "Arms control and disarmament",
                            hr = "Human rights",
                            co = "Colonialism",
                            ec = "Economic development"))
    
# 4.6. Summarize by country, year, and topic ----
    # Print votes_tidied
    print(votes_tidied)
    
    # Summarize the percentage "yes" per country-year-topic
    by_country_year_topic <- votes_tidied %>%
      group_by(country, year, topic) %>%
      summarize(total = n(), percent_yes = mean(vote == 1)) %>% 
      ungroup()
    
    # Print by_country_year_topic
    print(by_country_year_topic)
    
# 4.7. Visualizing trends in topics for one country ----
    # Load the ggplot2 package
    library(ggplot2)
    
    # Filter by_country_year_topic for just the US
    US_by_country_year_topic <- by_country_year_topic %>%
      filter(country == "United States")
    
    # Plot % yes over time for the US, faceting by topic
    ggplot(US_by_country_year_topic, aes(x = year, y = percent_yes)) +
      geom_line() +
      facet_wrap(~topic)
    
    
# 4.8. Nesting by topic and country ----
    # Load purrr, tidyr, and broom
    library(purrr)
    library(tidyr)
    library(broom)
    
    # Print by_country_year_topic
    print(by_country_year_topic)
    
    # Fit model on the by_country_year_topic dataset
    country_topic_coefficients <- by_country_year_topic %>%
      nest(-country, -topic) %>%
      mutate(model = map(data, ~ lm(percent_yes ~ year, data = .)),
             tidied = map(model, tidy)) %>%
      unnest(tidied)
    
    # Print country_topic_coefficients
    print(country_topic_coefficients)

        
# 4.9. Interpreting tidy models ----
    # Create country_topic_filtered
    country_topic_filtered <- country_topic_coefficients %>%
      filter(term == "year") %>%
      mutate(p.adjusted = p.adjust(p.value)) %>%
      filter(p.adjusted < .05)
    
# 4.10. Checking models visually ----
    # Create vanuatu_by_country_year_topic
    vanuatu_by_country_year_topic <- by_country_year_topic %>%
      filter(country == "Vanuatu")
    
    # Plot of percentage "yes" over time, faceted by topic
    ggplot(vanuatu_by_country_year_topic, aes(x = year, y = percent_yes)) + 
      geom_line() + 
      facet_wrap(~topic)