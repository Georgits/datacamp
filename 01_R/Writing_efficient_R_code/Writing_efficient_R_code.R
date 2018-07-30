# matrix nutzen statt data.frame
# rowSums nutzen: Faktor 4
# Use && instead of & : Facktor 2

library(readr)
library(microbenchmark)
library(parallel)

# Chapter 1: The Art of Benchmarking ----

# R version ----
# Print the R version details using version
version

# Assign the variable `major` to the major component
major <- version$major 

# Assign the variable `minor` to the minor component
minor <- version$minor


# Comparing read times of CSV and RDS files ----
# How long does it take to read movies from CSV?
system.time(read.csv("movies.csv"))

# How long does it take to read movies from RDS?
system.time(readRDS("movies.rds"))




# Elapsed time ----
# Load the microbenchmark package
# write.csv(movies, file = "movies.csv", quote = TRUE, row.names = FALSE)
# save.image("C:/Users/d91067/Desktop/datacamp/Writing_efficient_R_code/movies_csv.RData")
library(microbenchmark)

# define function
readData <- function(file){
  # Einlesen des Datensatzes
  read_delim("movies.csv",
             delim = ",",
             col_names = TRUE,
             col_types = cols(.default = "c"), # Spaltenformat: "character"
             locale = locale(
               encoding = "UTF-8",
               decimal_mark = ","
             )
  )
  }



# Compare the three functions
compare <- microbenchmark(read.csv("movies.csv"),
                          readData("movies.csv"),
                          load("movies_csv.RData"),
                          readRDS("movies.rds"),
                          times = 100)

# Print compare
compare




# Relative time ----
# Approximately, how much slower is the mean time of read.csv() compared to readRDS()?
608.04528 / 74.40677




# DataCamp hardware ----
# Load the benchmarkme package
library(benchmarkme)

# Assign the variable `ram` to the amount of RAM on this machine
ram <- get_ram()
ram

# Assign the variable `cpu` to the cpu specs
cpu <- get_cpu()
cpu



# Benchmark DataCamp's machine ---
# Load the package
library(benchmarkme)

# Run the io benchmark
res <- benchmark_io(runs = 1, size = 5)

# Plot the results
plot(res)





# Chapter 2: Fine Tuning: Efficient Base R -----
# Timings - growing a vector ----
n <- 30000
# Slow code
growing <- function(n) {
  x <- NULL
  for(i in 1:n)
    x <- c(x, rnorm(1))
  x
}

# Use `<-` with system.time() to store the result as res_grow
system.time(res_grow <- growing(n = 30000))




# Timings - pre-allocation ----
n <- 30000
# Fast code
pre_allocate <- function(n) {
  x <- numeric(n) # Pre-allocate
  for(i in 1:n) 
    x[i] <- rnorm(1)
  x
}

# Use `<-` with system.time() to store the result as res_allocate
n <- 30000
system.time(res_allocate <- pre_allocate(n = 30000))



# Vectorized code: multiplication ----
# The following piece of code is written like traditional C or Fortran code. 
# Instead of using the vectorized version of multiplication, it uses a for loop.

x <- rnorm(10)
x2 <- numeric(length(x))
for(i in 1:10)
  x2[i] <- x[i] * x[i]

# Your job is to make this code more "R-like" by vectorizing it.
# Store your answer as x2_imp
x2_imp <- x * x


# Vectorized code: calculating a log-sum ----
# A common operation in statistics is to calculate the sum of log probabilites.
# The following code calculates the log-sum (the sum of the logs). 
# However this piece of code could be significantly improved using vectorized code.
# Initial code
n <- 100
total <- 0
x <- runif(n)
for(i in 1:n) 
  total <- total + log(x[i])

# Rewrite in a single line. Store the result in log_sum
log_sum <- sum(log(x))




# Data frames and matrices - column selection ----
mat <- matrix(replicate(1000,rnorm(100)), nrow = 100)
df <- data.frame(replicate(1000,rnorm(100)))

# Which is faster, mat[, 1] or df[, 1]? 
microbenchmark(mat[, 1], df[, 1])


# Row timings ----
microbenchmark(mat[1,], df[1,])




# Profvis in action ----
# Load the data set
data(movies, package = "ggplot2movies") 

# Load the profvis package
library(profvis)

# Profile the following code with the profvis function
profvis(
  {
    # Load and select data
    movies <- movies[movies$Comedy == 1, ]
    
    # Plot data of interest
    plot(movies$year, movies$rating)
    
    # Loess regression line
    model <- loess(rating ~ year, data = movies)
    j <- order(movies$year)
    
    # Add a fitted line to the plot
    lines(movies$year[j], model$fitted[j], col = "red")
  })     ## Remember the closing brackets!






# Change the data frame to a matrix ----
# Load the microbenchmark package
library(microbenchmark)

# The previous data frame solution is defined
# d() Simulates 6 dices rolls
d <- function() {
  data.frame(
    d1 = sample(1:6, 3, replace = TRUE),
    d2 = sample(1:6, 3, replace = TRUE)
  )
}

# Complete the matrix solution
m <- function() {
  matrix(sample(1:6, 6, replace = TRUE), ncol = 2)
}

# Use microbenchmark to time m() and d()
microbenchmark(
  data.frame_solution = d(),
  matrix_solution     = m()
)




# Calculating row sums ----
# Example data
rolls <- m()

# Define the previous solution 
app <- function(x) {
  apply(x, 1, sum)
}

# Define the new solution
r_sum <- function(x) {
  rowSums(x)
}

# Compare the methods
microbenchmark(
  app_sol = app(rolls),
  r_sum_sol = r_sum(rolls)
)



# Use && instead of & ----
# Example data
is_double

# Define the previous solution
move <- function(is_double) {
  if (is_double[1] & is_double[2] & is_double[3]) {
    current <- 11 # Go To Jail
  }
}

# Define the improved solution
improved_move <- function(is_double) {
  if (is_double[1] && is_double[2] && is_double[3]) {
    current <- 11 # Go To Jail
  }
}

## microbenchmark both solutions
microbenchmark(move(is_double), improved_move(is_double), times = 1e5)





# Chapter 4: Turbo Charged Code: Parallel Programming ----
# How many cores does this machine have? ----
# Load the parallel package
library(parallel)

# Store the number of cores in the object no_of_cores
no_of_cores <- detectCores()

# Print no_of_cores
no_of_cores



# Moving to parApply ----
# To run code in parallel using the parallel package, the basic workflow has three steps.
# Create a cluster using makeCluster().
# Do some work.
# Stop the cluster using stopCluster().

  df <- data.frame(replicate(10000,sample(c(0,1), replace = TRUE, size = 1000)))
  # mat <- matrix(replicate(10000,sample(c(0,1), replace = TRUE, size = 10000)), nrow = 10000)
  mat <- matrix(replicate(10000,rbinom(10000,1,0.01)), nrow = 10000)
  
  dd <- mat


profvis::profvis({

  apply(dd, 2, median)

  # Determine the number of available cores
  no_cores <- detectCores()
  
  # Create a cluster via makeCluster
  cl <- makeCluster(no_cores - 1)
  
  # Parallelize this code
  parApply(cl, dd, 2, median)
  
  # Stop the cluster
  stopCluster(cl)
}
)

microbenchmark(
  apply(dd, 2, median),
  parApply(cl, dd, 2, median)
)



# Using parSapply() ----
# define function 
play <- function() {
  total <- no_of_rolls <- 0
  while(total < 10) {
    total <- total + sample(1:6, 1)
    
    # If even. Reset to 0
    if(total %% 2 == 0) total <- 0 
    no_of_rolls <- no_of_rolls + 1
  }
  no_of_rolls
}

# simulate game 100 times
res <- sapply(1:100, function(i) play())

# Parallelize
# Create a cluster via makeCluster (2 cores)
cl <- makeCluster(2)

# Export the play() function to the cluster
clusterExport(cl, "play")

# Parallelize this code
res <- parSapply(cl, 1:100, function(i) play())

# Stop the cluster
stopCluster(cl)




# Timings parSapply() ----
# Set the number of games to play
no_of_games <- 1e5

## Time serial version
system.time(serial <- sapply(1:no_of_games, function(i) play()))

## Set up cluster
cl <- makeCluster(4)
clusterExport(cl, "play")

## Time parallel version
system.time(par <- parSapply(cl, 1:no_of_games, function(i) play()))

## Stop cluster
stopCluster(cl)
