library(purrr)
library(readr)

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