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



