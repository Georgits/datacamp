library(gstat)
library(automap)
ca_geo <- readRDS("ca_geo.rds")
geo_bounds <- readRDS("ca_geo_bounds.rds")
preston_crime <- readRDS("pcrime-spatstat.rds")
preston_osm <- readRDS("osm_preston_gray.rds")



# Chapter 1. Introduction
# Simple spatial principles ----
# The number of points to create
n <- 200

# Set the range
xmin <- 0
xmax <- 1
ymin <- 0
ymax <- 2

# Sample from a Uniform distribution
x <- runif(n, xmin, xmax)
y <- runif(n, ymin, ymax)



# Plotting areas ----
# See pre-defined variables
ls.str()

# Plot points and a rectangle

mapxy <- function(a = NA){
  plot(x, y, asp = a)
  rect(xmin, ymin, xmax, ymax)
}

mapxy(a = 1)




# Uniform in a circle ----
# Load the spatstat package
library(spatstat)

# Create this many points, in a circle of this radius
n_points <- 300
radius <- 10

# Generate uniform random numbers up to radius-squared
r_squared <- runif(n_points, 0, radius^2)
angle <- runif(n_points, 0, 2*pi)

# Take the square root of the values to get a uniform spatial distribution
x <- sqrt(r_squared) * cos(angle)
y <- sqrt(r_squared) * sin(angle)

plot(disc(radius)); points(x, y)






# Quadrat count test for uniformity ----
# Some variables have been pre-defined
ls.str()

# Set coordinates and window
ppxy <- ppp(x = x, y = y, window = disc(radius))

# Test the point pattern
qt <- quadrat.test(ppxy)

# Inspect the results
plot(qt)
print(qt)




# Creating a uniform point pattern with spatstat ----
# Create a disc of radius 10
disc10 <- disc(10)

# Compute the rate as count divided by area
lambda <- 500 / area(disc10)

# Create a point pattern object
ppois <- rpoispp(lambda = lambda, win = disc10)

# Plot the Poisson point pattern
plot(ppois)




# Simulating clustered and inhibitory patterns ----
# Create a disc of radius 10
disc10 <- disc(10)

# Generate clustered points from a Thomas process
set.seed(123)
p_cluster <- rThomas(kappa = 0.35, scale = 1, mu = 3, win = disc10)
plot(p_cluster)

# Run a quadrat test
  quadrat.test(p_cluster, alternative = "clustered")

# Regular points from a Strauss process
set.seed(123)
p_regular <- rStrauss(beta = 2.9, gamma = 0.025, R = .5, W = disc10)
plot(p_regular)

# Run a quadrat test
quadrat.test(p_regular, alternative = "regular")



# Nearest-neighbor distributions ----
# Point patterns are pre-defined
p_poisson; p_regular

# Calc nearest-neighbor distances for Poisson point data
nnd_poisson <- nndist(p_poisson)

# Draw a histogram of nnds
hist(nnd_poisson)

# Estimate G(r)
G_poisson <- Gest(p_poisson)

# Plot G(r) vs. r
plot(G_poisson)

# Repeat for regular point data
nnd_regular <- nndist(p_regular)
hist(nnd_regular)
G_regular <- Gest(p_regular)
plot(G_regular)





# Other point pattern distribution functions ----
# Point patterns are pre-defined
p_poisson; p_cluster; p_regular

# Estimate the K-function for the Poisson points
K_poisson <- Kest(p_poisson, correction = "border")

# The default plot shows quadratic growth
plot(K_poisson, . ~ r)

# Subtract pi * r ^ 2 from the Y-axis and plot
plot(K_poisson, . - pi * r ^ 2 ~ r)

# Compute envelopes of K under random locations
K_cluster_env <- envelope(p_cluster, Kest, correction = "border")

# Insert the full formula to plot K minus pi * r^2
plot(K_cluster_env, . - pi * r ^ 2 ~ r)

# Repeat for regular data
K_regular_env <- envelope(p_regular, Kest, correction = "border")
plot(K_regular_env, . - pi * r ^ 2 ~ r)




# Tree location pattern ----
quadrat.test(redoak)






# Chapter 2. Point Pattern Analysis ----
# Crime in Preston ----
# Load the spatstat package
library(spatstat)
library(raster)

# Get some summary information on the dataset
summary(preston_crime)

# Get a table of marks
table(marks(preston_crime))

# Define a function to create a map
preston_map <- function(cols = c("green","red"), cex = c(1, 1), chars = c(1, 1)) {
  plotRGB(preston_osm) # from the raster package
  plot(preston_crime, cols = cols, chars = chars, cex = cex, add = TRUE, show.window = TRUE)
}

# Draw the map with colors, sizes and plot character
preston_map(
  cols = c("black", "red"), 
  cex = c(0.5, 1), 
  chars = c(19,19)
)




# Violent crime proportion estimation ----
# preston_crime has been pre-defined
preston_crime

# Use the split function to show the two point patterns
crime_splits <- split(preston_crime)

# Plot the split crime
plot(crime_splits)

# Compute the densities of both sets of points
crime_densities <- density(crime_splits)

# Calc the violent density divided by the sum of both
frac_violent_crime_density <- crime_densities[[2]] / 
  (crime_densities[[1]] + crime_densities[[2]])

# Plot the density of the fraction of violent crime
plot(frac_violent_crime_density)




# Bandwidth selection ----
# The first step is to compute the optimal bandwidth for kernel smoothing under the segregation model. 
# spseg() will scan over a range of bandwidths and compute a test statistic using a cross-validation method. 
# The bandwidth that maximizes this test statistic is the one to use. The returned value from spseg() in this case is a list, 
# and its hcv element is the one the contains this best bandwidth.


library(spatialkernel)

# Scan from 500m to 1000m in steps of 50m
bw_choice <- spseg(
  preston_crime, 
  h = seq(500, 1000, by = 50),
  opt = 1)

# Plot the results and highlight the best bandwidth
plotcv(bw_choice); abline(v = bw_choice$hcv, lty = 2, col = "red")

# Print the best bandwidth
print(bw_choice$hcv)

# Now you know the optimal smoothing parameter, you can do some kernel smoothing simulations. 



# Segregation probabilities ----
# The second step is to compute the probabilities for violent and non-violent crimes as a smooth surface, 
# as well as the p-values for a point-wise test of segregation. This is done by calling spseg() with opt = 3 and 
# a fixed bandwidth parameter h.
# Set the correct bandwidth and run for 10 simulations only
seg10 <- spseg(
  pts = preston_crime, 
  h = 800,
  opt = 3,
  ntest = 10, 
  proc =FALSE)
# Plot the segregation map for violent crime
plotmc(seg10, "Violent crime")

# Plot seg, the result of running 1000 simulations
plotmc(seg, "Violent crime")



# Mapping segregation ----
# The seg object is a list with several components. The X and Y coordinates of the grid are stored in the $gridx and $gridy 
# elements. The probabilities of each class of data (violent or non-violent crime) are in a matrix element $p with 
# a column for each class. The p-value of the significance test is in a similar matrix element called $stpvalue. 
# Rearranging columns of these matrices into a grid of values can be done with R's matrix() function. From there you can 
# construct list objects with a vector $x of X-coordinates, $y of Y-coordinates, and $z as the matrix. You can then feed 
# this to image() or contour() for visualization.

# Inspect the structure of the spatial segregation object
str(seg)

# Get the number of columns in the data so we can rearrange to a grid
ncol <- length(seg$gridx)

# Rearrange the probability column into a grid
prob_violent <- list(x = seg$gridx,
                     y = seg$gridy,
                     z = matrix(seg$p[, "Violent crime"],
                                ncol = ncol))
image(prob_violent)

# Rearrange the p-values, but choose a p-value threshold
p_value <- list(x = seg$gridx,
                y = seg$gridy,
                z = matrix(seg$stpvalue[, "Violent crime"] < 0.05,
                           ncol = ncol))
image(p_value)

# Create a mapping function
segmap <- function(prob_list, pv_list, low, high){
  
  # background map
  plotRGB(preston_osm)
  
  # p-value areas
  image(pv_list, 
        col = c("#00000000", "#FF808080"), add = TRUE) 
  
  # probability contours
  contour(prob_list,
          levels = c(low, high),
          col = c("#206020", "red"),
          labels = c("Low", "High"),
          add = TRUE)
  
  # boundary window
  plot(Window(preston_crime), add = TRUE)
}

# Map the probability and p-value
segmap(prob_violent, p_value, 0.05, 0.15)




# Sasquatch data ----
# Get a quick summary of the dataset
summary(sasq)

# Plot unmarked points
plot(unmark(sasq))

# Plot the points using a circle sized by date
plot(sasq, which.marks = "date")


# Temporal pattern of bigfoot sightings ----
# Show the available marks
names(marks(sasq))

# Histogram the dates of the sightings, grouped by year
hist(marks(sasq)$date, "year", freq = TRUE)

# Plot and tabulate the calendar month of all the sightings
plot(table(marks(sasq)$month))

# Split on the month mark
saqs_by_month <- split(sasq, "month", un = TRUE)

# Plot monthly maps
plot(saqs_by_month)

# Plot smoothed versions of the above split maps
plot(density(saqs_by_month))


# Preparing data for space-time clustering ----
library(splancs)
library(spatstat)
# To do a space-time clustering test with stmctest() from the splancs package, you first need to convert parts 
# of your ppp object. Functions in splancs tend to use matrix data instead of data frames.

# Get a matrix of event coordinates
sasq_xy <- as.matrix(coords(sasq))

# Check the matrix has two columns
dim(sasq_xy)

# Get a vector of event times
sasq_t <- marks(sasq)$date

# Extract a two-column matrix from the ppp object
sasq_poly <- as.matrix(as.data.frame(Window(sasq)))
dim(sasq_poly)

# Set the time limit to 1 day before and 1 day after the range of times
tlimits <- range(sasq_t) + c(-1, 1)

# Scan over 400m intervals from 100m to 20km
s <- seq(100, 20000, by = 400)

# Scan over 14 day intervals from one week to 31 weeks
tm <- seq(7, 7 * 31, by = 14)




# Monte-carlo test of space-time clustering -----
# Any space-time clustering in a data set will be removed if you randomly rearrange the dates of the data points. 
# The stmctest() function computes a clustering test statistic for your data based on the space-time K-function - how many 
# points are within a spatial and temporal window of a point of the data.

# The output from stmctest() is a list with a single t0 which is the test statistic for your data, and a vector of t from 
# the simulations.
# Run 999 simulations 
sasq_mc <- stmctest(sasq_xy, sasq_t, sasq_poly, tlimits, s, tm, nsim = 999, quiet = TRUE)
names(sasq_mc)

# Histogram the simulated statistics and add a line at the data value
ggplot(data.frame(sasq_mc), aes(x = t)) +
  geom_histogram(binwidth = 1e13) +
  geom_vline(aes(xintercept = t0))

# Compute the p-value as the proportion of tests greater than the data
sum(sasq_mc$t > sasq_mc$t0) / 1000





# Chapter 3. Areal Statistics ----
# London EU referendum data ----
library(raster)
# See what information we have for each borough
summary(london_ref)

# Which boroughs voted to "Leave"?
london_ref$NAME[london_ref$Leave > london_ref$Remain]

# Plot a map of the percentage that voted "Remain"
spplot(london_ref, zcol = "Pct_Remain")



# Cartogram ----
# Use the cartogram and rgeos packages
library(cartogram)
library(rgeos)

# Make a scatterplot of electorate vs borough area
names(london_ref)
plot(london_ref$Electorate, gArea(london_ref, byid = TRUE))

# Make a cartogram, scaling the area to the electorate
carto_ref <- cartogram(london_ref, "Electorate")
plot(carto_ref)

# Check the linearity of the electorate-area plot
plot(carto_ref$Electorate, gArea(carto_ref, byid = TRUE))

# Make a fairer map of the Remain percentage
spplot(carto_ref, "Pct_Remain")



# Spatial autocorrelation test ----
# Use the spdep package
library(spdep)

# Make neighbor list
borough_nb <- poly2nb(london_ref)

# Get center points of each borough
borough_centers <- coordinates(london_ref)

# Show the connections
plot(london_ref); plot(borough_nb, borough_centers, add = TRUE)

# Map the total pop'n
spplot(london_ref, zcol = "TOTAL_POP")

# Run a Moran I test on total pop'n
moran.test(
  london_ref$TOTAL_POP, 
  nb2listw(borough_nb)
)

# Map % Remain
spplot(london_ref, zcol = "Pct_Remain")

# Run a Moran I MC test on % Remain
moran.mc(
  london_ref$Pct_Remain, 
  nb2listw(borough_nb), 
  nsim = 999
)





# London health data ----
library(sp)
# Get a summary of the data set
summary(london)

# Map the OBServed number of flu reports
spplot(london, "Flu_OBS")

# Compute the overall incidence of flu
r <- sum(london$Flu_OBS) / sum(london$TOTAL_POP)
r

# Calculate the expected number for each borough
london$Flu_EXP <- london$TOTAL_POP * r

# Calculate the ratio of OBServed to EXPected
london$Flu_SMR <- london$Flu_OBS / london$Flu_EXP

# Map the SMR
spplot(london, "Flu_SMR")




# Binomial confidence intervals ----
library(sp)
library(ggplot)

# For the binomial statistics function
library(epitools)

# Get CI from binomial distribution
flu_ci <- binom.exact(london$Flu_OBS, london$TOTAL_POP)

# Add borough names
flu_ci$NAME <- london$NAME

# Calculate London rate, then compute SMR
r <- sum(london$Flu_OBS) / sum(london$TOTAL_POP)
flu_ci$SMR <- flu_ci$proportion / r

# Subset the high SMR data
flu_high <- flu_ci[flu_ci$SMR > 1, ]

# Plot estimates with CIs
library(ggplot2)
ggplot(flu_high, aes(x = NAME, y = proportion / r,
                     ymin = lower / r, ymax = upper / r)) +
  geom_pointrange() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))





# Exceedence probabilities ----
# Probability of a binomial exceeding a multiple
binom.exceed <- function(observed, population, expected, e){
  1 - pbinom(e * expected, population, prob = observed / population)
}

# Compute P(rate > 2)
london$Flu_gt_2 <- binom.exceed(
  observed = london$Flu_OBS,
  population = london$TOTAL_POP,
  expected = london$Flu_EXP,
  e = 2)

# Use a 50-color palette that only starts changing at around 0.9
pal <- c(
  rep("#B0D0B0", 40),
  colorRampPalette(c("#B0D0B0", "orange"))(5), 
  colorRampPalette(c("orange", "red"))(5)
)

# Plot the P(rate > 2) map
spplot(london, "Flu_gt_2", col.regions = pal, at = seq(0, 1, len = 50))





# A Poisson GLM ----
# Fit a poisson GLM.
# To cope with count data coming from populations of different sizes, you specify an offset argument. 
# This adds a constant term for each row of the data in the model. The log of the population is used in the offset term.
model_flu <- glm(
  Flu_OBS ~ HealthDeprivation, 
  offset = log(TOTAL_POP), 
  data = london, 
  family = poisson)

# Is HealthDeprivation significant?
summary(model_flu)

# Put residuals into the spatial data.
london$Flu_Resid <- residuals(model_flu)

# Map the residuals using spplot
spplot(london, "Flu_Resid")



# Residuals ----
# Compute the neighborhood structure.
library(spdep)
borough_nb <- poly2nb(london)

# Test spatial correlation of the residuals.
moran.mc(london$Flu_Resid, listw = nb2listw(borough_nb), nsim = 999)




# Fit a Bayesian GLM ----
# Use R2BayesX
library(R2BayesX)

# Fit a GLM
model_flu <- glm(Flu_OBS ~ HealthDeprivation, offset = log(TOTAL_POP),
                 data = london, family = poisson)

# Summarize it                    
summary(model_flu)

# Calculate coeff confidence intervals
confint(model_flu)

# Fit a Bayesian GLM
# he syntax for bayesx() is similar, but the offset has to be specified explicitly from the data frame, 
# the family name is in quotes, and the spatial data frame needs to be turned into a plain data frame.
bayes_flu <- bayesx(Flu_OBS ~ HealthDeprivation, offset = log(london$TOTAL_POP), 
                    family = "poisson", data = data.frame(london), 
                    control = bayesx.control(seed = 17610407))

# Summarize it                    
# Plot the samples from the Bayesian model. On the left is the "trace" of samples in sequential order, 
# and on the right is the parameter density. For this model there is an intercept and a slope for the 
# Health Deprivation score. The parameter density should correspond with the parameter summary.
summary(bayes_flu)

# Look at the samples from the Bayesian model
plot(samples(bayes_flu))





# Adding a spatially autocorrelated effect ----
# Compute adjacency objects
borough_nb <- poly2nb(london)
borough_gra <- nb2gra(borough_nb)

# Fit spatial model
flu_spatial <- bayesx(
  Flu_OBS ~ HealthDeprivation + sx(i, bs = "spatial", map = borough_gra),
  offset = log(london$TOTAL_POP),
  family = "poisson", data = data.frame(london), 
  control = bayesx.control(seed = 17610407)
)

# Summarize the model
summary(flu_spatial)




# Mapping the spatial effects ----
# Summarise the model
summary(flu_spatial)

# Map the fitted spatial term only
london$spatial <- fitted(flu_spatial, term = "sx(i):mrf")[, "Mean"]
spplot(london, zcol = "spatial")

# Map the residuals
london$spatial_resid <- residuals(flu_spatial)[, "mu"]
spplot(london, zcol = "spatial_resid")

# Test residuals for spatial correlation
moran.mc(london$spatial_resid, nb2listw(borough_nb), 999)



# Chapter 4.  Geostatistics ----
# Canadian geochemical survey data ----
# ca_geo has been pre-defined
str(ca_geo, 1)

# See what measurements are at each location
names(ca_geo)

# Get a summary of the acidity (pH) values
summary(ca_geo$pH)

# Look at the distribution
hist(ca_geo$pH)

# Make a vector that is TRUE for the missing data
miss <- is.na(ca_geo$pH)
table(miss)

# Plot a map of acidity
spplot(ca_geo[!is.na(ca_geo$pH), ], "pH")




# Fitting a trend surface ----
# ca_geo has been pre-defined
str(ca_geo, 1)

# Are they called lat-long, up-down, or what?
coordnames(ca_geo)

# Complete the formula
m_trend <- lm(pH ~ x + y, as.data.frame(ca_geo))

# Check the coefficients
summary(m_trend)



# Predicting from a trend surface ----
# The acidity survey data, ca_geo, and the linear model, m_trend have been pre-defined.
# ca_geo, miss, m_trend have been pre-defined
ls.str()

# Make a vector that is TRUE for the missing data
miss <- is.na(ca_geo$pH)

# Create a data frame of missing data
ca_geo_miss <- as.data.frame(ca_geo)[miss, ]

# Predict pH for the missing data
predictions <- predict(m_trend, newdata = ca_geo_miss, se.fit = TRUE)

# Compute the exceedence probability
pAlkaline <- 1 - pnorm(7, mean = predictions$fit, sd = predictions$se.fit)
hist(pAlkaline)



# Variogram estimation ----
# ca_geo, miss have been pre-defined
ls.str()

# Make a cloud from the non-missing data up to 10km
plot(variogram(pH ~ 1, ca_geo[!miss, ], cloud = TRUE, cutoff = 10000))

# Make a variogram of the non-missing data
plot(variogram(pH ~ 1, ca_geo[!miss, ]))



# Variogram with spatial trend ----
# ca_geo, miss have been pre-defined
ls.str()

# See what coordinates are called
coordnames(ca_geo)

# The pH depends on the coordinates
ph_vgm <- variogram(pH ~ x + y, ca_geo[!miss, ])
plot(ph_vgm)




# Variogram model fitting ----

# The nugget is the value of the semivariance at zero distance.
# The partial sill, psill is the difference between the sill and the nugget.
# Set the range to the distance at which the variogram has got about half way between the nugget and the sill.
# ca_geo, miss, ph_vgm have been pre-defined
ls.str()

# Eyeball the variogram and estimate the initial parameters
nugget <- 0.16
psill <- 0.13
range <- 10000

# Fit the variogram
v_model <- fit.variogram(
  ph_vgm, 
  model = vgm(
    model = "Ste",
    nugget = nugget,
    psill = psill,
    range = range,
    kappa = 0.5
  )
)

# Show the fitted variogram on top of the binned variogram
plot(ph_vgm, model = v_model)
print(v_model)





# Filling in the gaps ----
# ca_geo, miss, v_model have been pre-defined
ls.str()

# Set the trend formula and the new data
km <- krige(pH ~ x + y, ca_geo[!miss, ], newdata = ca_geo[miss, ], model = v_model)
names(km)

# Plot the predicted values
spplot(km, "var1.pred")

# Compute the probability of alkaline samples, and map
km$pAlkaline <- 1 - pnorm(7, mean = km$var1.pred, sd = sqrt(km$var1.var))
spplot(km, "pAlkaline")





# Making a prediction grid ----
# ca_geo, geo_bounds have been pre-defined
ls.str()

# Plot the polygon and points
plot(geo_bounds); points(ca_geo)

# Find the corners of the boundary
bbox(geo_bounds)

# Define a 2.5km square grid over the polygon extent. The first parameter is
# the bottom left corner.
grid <- GridTopology(c(537853, 5536290), c(2500, 2500), c(72, 48))

# Create points with the same coordinate system as the boundary
gridpoints <- SpatialPoints(grid, proj4string = CRS(projection(geo_bounds)))
plot(gridpoints)

# Crop out the points outside the boundary
cropped_gridpoints <- crop(gridpoints, geo_bounds)
plot(cropped_gridpoints)

# Convert to SpatialPixels
spgrid <- SpatialPixels(cropped_gridpoints)
coordnames(spgrid) <- c("x", "y")
plot(spgrid)




# Gridded predictions ----
# spgrid, v_model have been pre-defined
ls.str()

# Do kriging predictions over the grid
ph_grid <- krige(pH ~ x + y, ca_geo[!miss, ], newdata = spgrid, model = v_model)

# Calc the probability of pH exceeding 7
ph_grid$pAlkaline <- 1 - pnorm(7, mean = ph_grid$var1.pred, sd = sqrt(ph_grid$var1.pred))

# Map the probability of alkaline samples
spplot(ph_grid, zcol = "pAlkaline")



# Auto-kriging at point locations ----
# The autoKrige() function in the automap package computes binned variograms, fits models, does model selection, and performs kriging 
# by making multiple calls to the gstat functions you used previously. It can be a great time-saver but you should always check the results carefully.
# ca_geo, miss are pre-defined
miss <- is.na(ca_geo$pH)

ls.str()

# Kriging with linear trend, predicting over the missing points
ph_auto <- autoKrige(
  pH ~ x + y, 
  input_data = ca_geo[!miss, ], 
  new_data = ca_geo[miss, ], 
  model = "Mat"
)

# Plot the variogram, predictions, and standard error
plot(ph_auto)





# Auto-kriging over a grid ----
# ca_geo, miss, spgrid, ph_grid, v_model are pre-defined
ls.str()

# Auto-run the kriging
ph_auto_grid <- autoKrige(pH ~ x + y, input_data = ca_geo[!miss, ], new_data = spgrid)

# Remember predictions from manual kriging
plot(ph_grid)

# Plot predictions and variogram fit
plot(ph_auto_grid)

# Compare the variogram model to the earlier one
v_model
ph_auto_grid$var_model