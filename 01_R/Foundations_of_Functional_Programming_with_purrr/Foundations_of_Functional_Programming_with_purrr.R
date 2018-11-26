library(purrr)
library(readr)
library(repurrrsive)
library(ggplot2)

# Chapter 1: Simplifying Iteration and Lists With purrr

# Introduction to iteration
files = list.files(pattern="*.csv")
# Initialize list
all_files <- list()
# For loop to read files into a list
for(i in seq_along(files)){
  all_files[[i]] <- read_csv(file = files[[i]])
}
# Output size of list object
length(all_files)



# Iteration with purrr
# Use map to iterate
all_files_purrr <- map(files, read_csv) 
# Output size of list object
length(all_files_purrr)




# More iteration with for loops
list_of_df <- rep(list(c("1", "2", "3", "4")), 10)
# Check the class type of the first element
class(list_of_df[[1]])
# Change each element from a character to a number
for(i in seq_along(list_of_df)){
  list_of_df[[i]] <- as.numeric(list_of_df[[i]])
}
# Check the class type of the first element
class(list_of_df[[1]])
# Print out the list
list_of_df






# More iteration with purrr
list_of_df <- rep(list(c("1", "2", "3", "4")), 10)
# Check the class type of the first element
class(list_of_df[[1]])  
# Change each character element to a number
list_of_df <- map(list_of_df, as.numeric)
# Check the class type of the first element again
class(list_of_df[[1]])
# Print out the list
list_of_df


# Subsetting lists
# Load repurrrsive package, to get access to the wesanderson dataset
library(repurrrsive)
# Load wesanderson dataset
data(wesanderson)
# Get structure of first element in wesanderson
str(wesanderson[1])
# Get structure of GrandBudapest element in wesanderson
str(wesanderson$GrandBudapest)



# Subsetting list elements
# Third element of the first wesanderson vector
wesanderson[[1]][3]
# Fourth element of the GrandBudapest wesanderson vector
wesanderson$GrandBudapest[4]
# Subset the first element of the sw_films data
sw_films[[1]]
# Subset the first element of the sw_films data, title column 
sw_films[[1]]$title



# map() argument alternatives
# Map over wesanderson to get the length of each element
map(wesanderson, length)
# Map over wesanderson, and determine the length of each element
map(wesanderson, ~length(.x))



# map_*
# Map over wesanderson and determine the length of each element
map(wesanderson, length)
# Create a numcolors column and fill with length of each wesanderson element
data.frame(numcolors = map_dbl(wesanderson, ~length(.x)))







# Chapter 2: More complex iterations
# Names & pipe refresher
# Use pipes to check for names in sw_films
sw_films %>%
  names()


# Setting names
# Set names so each element of the list is named for the film title
sw_films_named <- sw_films %>% 
  set_names(map_chr(sw_films, "title"))

# Check to see if the names worked/are correct
names(sw_films_named)



# Pipes in map()
# Create a list of values from 1 through 10
numlist <- list(1,2,3,4,5,6,7,8,9,10)

# Iterate over the numlist 
map(numlist, ~.x %>% sqrt() %>% sin())




# Simulating Data with Purrr
# List of sites north, east, and west
sites <- list("north", "east", "west")

# Create a list of dataframes, each with a years, a, and b column 
list_of_df <-  map(sites,  
                   ~data.frame(sites = .x,
                               a = rnorm(mean = 5, n = 200, sd = (5/2)),
                               b = rnorm(mean = 200, n = 200, sd = 15)))

list_of_df




# Map over the models to look at the relationship of a vs b
list_of_df %>%
  map(~ lm(a ~ b, data = .)) %>%
  map(summary)



# map_chr()
# Pull out the director element of sw_films in a list and character vector
map(sw_films, ~.x[["director"]])
map_chr(sw_films, ~.x[["director"]])

# Compare outputs when checking if director is George Lucas
map(sw_films, ~.x[["director"]] == "George Lucas")
map_lgl(sw_films, ~.x[["director"]] == "George Lucas")




# map_dbl() and map_int()
# Pull out episode_id element as list
map(sw_films, ~.x[["episode_id"]])

# Pull out episode_id element as double vector
map_dbl(sw_films, ~.x[["episode_id"]])

# Pull out episode_id element as integer vector
map_int(sw_films, ~.x[["episode_id"]])




# Simulating data with multiple inputs using map2()
# List of 1 through 3
means <- list(1,2,3)
# Create sites list
sites <- list("north", "west", "east")
# Map over two arguments: sites and mu
list_of_files_map2 <- map2(sites, means, ~data.frame(sites = .x,
                                                     a = rnorm(mean = .y, n = 200, sd = (5/2))))
list_of_files_map2




# Simulating data 3+ inputs with pmap()
sites <- list("north", "west", "east")
means <- list(1,2,3)
means2 <- list(0.5,1,1.5)
sigma <- list(1,2,3)
sigma2 <- list(0.5,1,1.5)

# Create a master list, a list of lists
pmapinputs <- list(sites = sites,  means = means, sigma = sigma, 
                   means2 = means2, sigma2 = sigma2)

# Map over the master list
list_of_files_pmap <- pmap(pmapinputs, 
                           function(sites, means, sigma, means2, sigma2) 
                             data.frame(sites = sites,
                                        a = rnorm(mean = means, n = 200, sd = sigma),
                                        b = rnorm(mean = means2, n = 200, sd = sigma2)))

list_of_files_pmap






# Chapter 3: Troubleshooting lists with purrr
# safely() replace with NA

# Map safely over log
a <- list(-10, 1, 10, 0) %>% 
  map(safely(log, otherwise = NA_real_)) %>%
  # Transpose the result
  transpose()

# Print the list
a

# Print the result element in the list
a[["result"]]

# Print the error element in the list
a[["error"]]





# Convert data to numeric with purrr
# Load sw_people data
data(sw_people)

# Map over sw_people and pull out the height element
height_cm <- map(sw_people, ~.x[["height"]]) %>%
  map(function(x){
    ifelse(x == "unknown",NA,
           as.numeric(x))
  })




# Finding the problem areas
# Map over sw_people and pull out the height element
height_ft <- map(sw_people , ~.x[["height"]]) %>% 
  map(safely(function(x){
    x * 0.0328084
  }, quiet = FALSE)) %>% 
  transpose()

# Print your list, the result element, and the error element
height_ft
height_ft[["result"]]
height_ft[["error"]]





# Replace safely() with possibly()
# Take the log of each element in the list
a <- list(-10, 1, 10, 0) %>% 
  map(possibly(function(x){
    log(x)
  },otherwise = NA_real_))



# Convert values with possibly()
# Create a piped workflow that returns double vectors
height_cm %>%  
  map_dbl(possibly(function(x){
    # Convert centimeters to feet
    x * 0.0328084
  }, otherwise = NA_real_))



# Comparing walk() vs no walk() outputs
people_by_film <- sw_people[[1]]

# Print normally
people_by_film
# Print with walk
walk(people_by_film, print)




# walk() for printing cleaner list outputs
# Load the gap_split data
data(gap_split)

# Map over the first 10 elements of gap_split
plots <- map2(gap_split[1:10], 
              names(gap_split[1:10]), 
              ~ ggplot(.x, aes(year, lifeExp)) + 
                geom_line() +
                labs(title = .y))

# Object name, then function name
# Load the gap_split data
data(gap_split)

# Map over the first 10 elements of gap_split
plots <- map2(gap_split[1:10], 
              names(gap_split[1:10]), 
              ~ ggplot(.x, aes(year, lifeExp)) + 
                geom_line() +
                labs(title = .y))

# Object name, then function name
walk(plots, print)