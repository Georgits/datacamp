load("Income.RData")
income_train <- incometrain
income_test <- incometest

unemployment <- readRDS("unemployment.rds")
bloodpressure <- readRDS("bloodpressure.rds")
houseprice <- readRDS("houseprice.rds")
bikes <- load("bikes.RData")
Soybean <- load("Soybean.RData")
dframe <- read.csv("dframe.csv", sep = ";",  row.names = 1)
testframe <- read.csv("testframe.csv", sep = ";",  row.names = 1)
sparrow <- readRDS("sparrow.rds")

library(broom)
library(sigr)
library(ggplot2)
library(WVPlots)
library(vtreat)
library(Sleuth3)
library(dplyr)
library(tidyr)
library(broom)
library(mgcv)
library(ranger)
library(vtreat)
library(magrittr)
library(xgboost)


# 1. Chapter : Whats is a regression? ----
# Code a simple one-variable regression ----
# unemployment is loaded in the workspace
summary(unemployment)

# Define a formula to express female_unemployment as a function of male_unemployment
fmla <- female_unemployment ~ male_unemployment

# Print it
fmla

# Use the formula to fit a model: unemployment_model
unemployment_model <- lm(fmla, data = unemployment)

# Print it
unemployment_model






# Examining a model ----
# broom and sigr are already loaded in your workspace
# Print unemployment_model
unemployment_model

# Call summary() on unemployment_model to get more details
summary(unemployment_model)

# Call glance() on unemployment_model to see the details in a tidier form
glance(unemployment_model)

# Call wrapFTest() on unemployment_model to see the most relevant details
wrapFTest(unemployment_model)




# Predicting from the unemployment model ----
# unemployment is in your workspace
summary(unemployment)

# newrates is in your workspace
newrates <- data.frame(male_unemployment  = c(5))
newrates

# Predict female unemployment in the unemployment data set
unemployment$prediction <-  predict(unemployment_model, newdata = unemployment)

# Make a plot to compare predictions to actual (prediction on x axis). 
ggplot(unemployment, aes(x = prediction, y = female_unemployment)) + 
  geom_point() +
  geom_abline(color = "blue")

# Predict female unemployment rate when male unemployment is 5%
pred <- predict(unemployment_model, newdata = newrates)
# Print it
pred




# Multivariate linear regression (Part 1) ----
# bloodpressure is in the workspace
summary(bloodpressure)

# Create the formula and print it
fmla <- blood_pressure ~ age + weight
fmla

# Fit the model: bloodpressure_model
bloodpressure_model <- lm(fmla, data = bloodpressure)

# Print bloodpressure_model and call summary() 
bloodpressure_model
summary(bloodpressure_model)




# Multivariate linear regression (Part 2) ----
# predict blood pressure using bloodpressure_model :prediction
bloodpressure$prediction <- predict(bloodpressure_model, newdata = bloodpressure)

# plot the results
ggplot(bloodpressure, aes(x = prediction, y = blood_pressure)) + 
  geom_point() +
  geom_abline(color = "blue")
  






# Chapter 2: Training and Evaluating Regression ----
# Graphically evaluate the unemployment model ----
# unemployment is in the workspace
summary(unemployment)

# unemployment_model is in the workspace
summary(unemployment_model)

# Make predictions from the model
unemployment$predictions <- predict(unemployment_model, newdata = unemployment)

# Fill in the blanks to plot predictions (on x-axis) versus the female_unemployment rates
ggplot(unemployment, aes(x = predictions, y = female_unemployment)) + 
  geom_point() + 
  geom_abline()

# Calculate residuals
unemployment$residuals <- unemployment$female_unemployment - unemployment$predictions

# Fill in the blanks to plot predictions (on x-axis) versus the residuals
ggplot(unemployment, aes(x = predictions, y = residuals)) + 
  geom_pointrange(aes(ymin = 0, ymax = residuals)) + 
  geom_hline(yintercept = 0, linetype = 3) + 
  ggtitle("residuals vs. linear model prediction")




# The gain curve to evaluate the unemployment model ----
# Plot the Gain Curve
GainCurvePlot(unemployment, "prediction", "female_unemployment", "Unemployment model")


# Calculate RMSE ----
# For convenience put the residuals in the variable res
res <- unemployment$residuals

# Calculate RMSE, assign it to the variable rmse and print it
(rmse <- sqrt(mean(res^2)))

# Calculate the standard deviation of female_unemployment and print it
(sd_unemployment <- sd(unemployment$female_unemployment))



# Calculate R-Squared ----
# Calculate mean female_unemployment: fe_mean. Print it
(fe_mean <- mean(unemployment$female_unemployment))

# Calculate total sum of squares: tss. Print it
(tss <- sum((unemployment$female_unemployment - fe_mean)^2))

# Calculate residual sum of squares: rss. Print it
(rss <- sum(unemployment$residuals^2))

# Calculate R-squared: rsq. Print it. Is it a good fit?
(rsq <- 1 - rss / tss)

# Get R-squared from glance. Print it
(rsq_glance <- glance(unemployment_model)$r.squared)



# Correlation and R-squared ----
# Get the correlation between the prediction and true outcome: rho and print it
(rho <- cor(unemployment$female_unemployment, unemployment$predictions))

# Square rho: rho2 and print it
(rho2 <- rho^2)

# Get R-squared from glance and print it
(rsq_glance <- glance(unemployment_model)$r.squared)





# Generating a random test/train split ----
# mpg is in the workspace
summary(mpg)
dim(mpg)

# Use nrow to get the number of rows in mpg (N) and print it
(N <- nrow(mpg))

# Calculate how many rows 75% of N should be and print it
# Hint: use round() to get an integer
(target <- round(0.75 * N, 0))

# Create the vector of N uniform random variables: gp
gp <- runif(N)

# Use gp to create the training set: mpg_train (75% of data) and mpg_test (25% of data)
mpg_train <- mpg[gp < 0.75,]
mpg_test <- mpg[gp >= 0.75,]

# Use nrow() to examine mpg_train and mpg_test
nrow(mpg_train)
nrow(mpg_test)





# Train a model using test/train split ----
# mpg_train is in the workspace
summary(mpg_train)

# Create a formula to express cty as a function of hwy: fmla and print it.
(fmla <- cty ~ hwy)

# Now use lm() to build a model mpg_model from mpg_train that predicts cty from hwy 
mpg_model <- lm(fmla, data = mpg_train)

# Use summary() to examine the model
summary(mpg_model)



# Evaluate a model using test/train split ----
# Functions rmse() and r_squared() to calculate RMSE and R-squared have been provided for convenience:
rmse <- function(predcol, ycol) {
  res = predcol-ycol
  sqrt(mean(res^2))
}


r_squared <- function(predcol, ycol) {
  tss = sum( (ycol - mean(ycol))^2 )
  rss = sum( (predcol - ycol)^2 )
  1 - rss/tss
}

# predict cty from hwy for the training set
mpg_train$pred <- predict(mpg_model, newdata = mpg_train)

# predict cty from hwy for the test set
mpg_test$pred <- predict(mpg_model, newdata = mpg_test)

# Evaluate the rmse on both training and test data and print them
(rmse_train <- rmse(mpg_train$pred, mpg_train$cty))
(rmse_test <- rmse(mpg_test$pred, mpg_test$cty))


# Evaluate the r-squared on both training and test data.and print them
(rsq_train <- r_squared(mpg_train$pred, mpg_train$cty))
(rsq_test <- r_squared(mpg_test$pred, mpg_test$cty))

# Plot the predictions (on the x-axis) against the outcome (cty) on the test data
ggplot(mpg_test, aes(x = pred, y = cty)) + 
  geom_point() + 
  geom_abline()





# Create a cross validation plan ----
# kWayCrossValidation() creates a cross validation plan with the following call:
  
  # splitPlan <- kWayCrossValidation(nRows, nSplits, dframe, y)

# where nRows is the number of rows of data to be split, and nSplits is the desired number of cross-validation folds.
# Strictly speaking, dframe and y aren't used by kWayCrossValidation; they are there for compatibility 
# with other vtreat data partitioning functions. You can set them both to NULL.

# The resulting splitPlan is a list of nSplits elements; each element contains two vectors:

# train: the indices of dframe that will form the training set
# app: the indices of dframe that will form the test (or application) set

# mpg is in the workspace
summary(mpg)

# Get the number of rows in mpg
nRows <- nrow(mpg)

# Implement the 3-fold cross-fold plan with vtreat
k <- 3
splitPlan <- kWayCrossValidation(nRows, k, NULL, NULL)

# Examine the split plan
str(splitPlan)

# !!! die app-Vektoren haben keine Überschneidungen, deswegen funktioniert die Schleife unten.
# intersect(splitPlan[[1]][["app"]],splitPlan[[2]][["app"]])


# Evaluate a modeling procedure using n-fold cross-validation
# mpg is in the workspace
summary(mpg)

# splitPlan is in the workspace
str(splitPlan)

# Run the 3-fold cross validation plan from splitPlan
k <- 3 # Number of folds
mpg$pred.cv <- 0 
for(i in 1:k) {
  split <- splitPlan[[i]]
  model <- lm(cty ~ hwy, data = mpg[split$train, ])
  mpg$pred.cv[split$app] <- predict(model, newdata = mpg[split$app,])
}

# Predict from a full model
mpg$pred <- predict(lm(cty ~ hwy, data = mpg))

# Get the rmse of the full model's predictions
rmse(mpg$pred, mpg$cty)

# Get the rmse of the cross-validation predictions
rmse(mpg$pred.cv, mpg$cty)






# Chapter 3: Issues to Consider ----
# Examining the structure of categorical inputs ----
# Call str on flowers to see the types of each column
str(flowers)

# Use unique() to see how many possible values Time takes
unique(flowers$Time)

# Build a formula to express Flowers as a function of Intensity and Time: fmla. Print it
(fmla <- as.formula("Flowers ~ Intensity + Time"))

# Use fmla and model.matrix to see how the data is represented for modeling
mmat <- model.matrix(fmla, data = flowers)

# Examine the first 20 lines of flowers
head(flowers, n = 20)

# Examine the first 20 lines of mmat
head(mmat, n = 20)



# Modeling with categorical inputs ----
# flowers in is the workspace
str(flowers)

# fmla is in the workspace
fmla

# Fit a model to predict Flowers from Intensity and Time : flower_model
flower_model <- lm(fmla, data = flowers)

# Use summary on mmat to remind yourself of its structure
summary(mmat)

# Use summary to examine flower_model 
summary(flower_model)

# Predict the number of flowers on each plant
flowers$predictions <- predict(flower_model, data = flowers)

# Plot predictions vs actual flowers (predictions on x-axis)
ggplot(flowers, aes(x = predictions, y = Flowers)) + 
  geom_point() +
  geom_abline(color = "blue") 






# Modeling an interaction ----
# alcohol is in the workspace
summary(alcohol)

# Create the formula with main effects only
(fmla_add <- Metabol ~ Gastric + Sex )

# Create the formula with interactions
(fmla_interaction <- Metabol ~ Gastric + Gastric : Sex )

# Fit the main effects only model
model_add <- lm(fmla_add, data = alcohol)

# Fit the interaction model
model_interaction <- lm(fmla_interaction, data = alcohol)

# Call summary on both models and compare
summary(model_add)
summary(model_interaction)








# Modeling an interaction (2) ----
# alcohol is in the workspace
summary(alcohol)

# Both the formulae are in the workspace
fmla_add
fmla_interaction

# Create the splitting plan for 3-fold cross validation
set.seed(34245)  # set the seed for reproducibility
splitPlan <- kWayCrossValidation(nrow(alcohol), 3, NULL, NULL)

# Sample code: Get cross-val predictions for main-effects only model
alcohol$pred_add <- 0  # initialize the prediction vector
for(i in 1:3) {
  split <- splitPlan[[i]]
  model_add <- lm(fmla_add, data = alcohol[split$train, ])
  alcohol$pred_add[split$app] <- predict(model_add, newdata = alcohol[split$app, ])
}

# Get the cross-val predictions for the model with interactions
alcohol$pred_interaction <- 0 # initialize the prediction vector
for(i in 1:3) {
  split <- splitPlan[[i]]
  model_interaction <- lm(fmla_interaction, data = alcohol[split$train, ])
  alcohol$pred_interaction[split$app] <- predict(model_interaction, newdata = alcohol[split$app, ])
}

# Get RMSE
alcohol %>% 
  gather(key = modeltype, value = pred, pred_add, pred_interaction) %>%
  mutate(residuals = Metabol - pred) %>%      
  group_by(modeltype) %>%
  summarize(rmse = sqrt(mean(residuals^2)))






# Relative error -----
# fdata is in the workspace
summary(fdata)

# Examine the data: generate the summaries for the groups large and small:
fdata %>% 
  group_by(label) %>%     # group by small/large purchases
  summarize(min  = min(y),   # min of y
            mean = mean(y),   # mean of y
            max  = max(y))   # max of y

# Fill in the blanks to add error columns
fdata2 <- fdata %>% 
  group_by(label) %>%       # group by label
  mutate(residual = pred -y,  # Residual
         relerr   = residual / y)  # Relative error

# Compare the rmse and rmse.rel of the large and small groups:
fdata2 %>% 
  group_by(label) %>% 
  summarize(rmse     = sqrt(mean(residual^2)),   # RMSE
            rmse.rel = sqrt(mean(relerr^2)))   # Root mean squared relative error

# Plot the predictions for both groups of purchases
ggplot(fdata2, aes(x = pred, y = y, color = label)) + 
  geom_point() + 
  geom_abline() + 
  facet_wrap(~ label, ncol = 1, scales = "free") + 
  ggtitle("Outcome vs prediction")





# Modeling log-transformed monetary output ----
# Examine Income2005 in the training set
summary(income_train$Income2005)

# Write the formula for log income as a function of the tests and print it
(fmla.log <- log(Income2005) ~ Arith + Word + Parag + Math + AFQT)

# Fit the linear model
model.log <-  lm(fmla.log, data = income_train)

# Make predictions on income_test
income_test$logpred <- predict(model.log, newdata = income_test)
summary(income_test$logpred)

# Convert the predictions to monetary units
income_test$pred.income <- exp(income_test$logpred)
summary(income_test$pred.income)

#  Plot predicted income (x axis) vs income
ggplot(income_test, aes(x = pred.income, y = Income2005)) + 
  geom_point() + 
  geom_abline(color = "blue")




# Comparing RMSE and root-mean-squared Relative Error ----
# fmla.abs is in the workspace
(fmla.abs <- Income2005 ~ Arith + Word + Parag + Math + AFQT)

# model.abs is in the workspace
model.abs <-  lm(fmla.abs, data = income_train)
summary(model.abs)

# Add predictions to the test set
income_test <- income_test %>%
  mutate(pred.absmodel = predict(model.abs, income_test),        # predictions from model.abs
         pred.logmodel = exp(predict(model.log, income_test)))   # predictions from model.log

# Gather the predictions and calculate residuals and relative error
income_long <- income_test %>% 
  gather(key = modeltype, value = pred, pred.absmodel, pred.logmodel) %>%
  mutate(residual = pred - Income2005,   # residuals
         relerr   = residual / Income2005)   # relative error

# Calculate RMSE and relative RMSE and compare
income_long %>% 
  group_by(modeltype) %>%      # group by modeltype
  summarize(rmse     = sqrt(mean(residual^2)),    # RMSE
            rmse.rel = sqrt(mean(relerr^2)))    # Root mean squared relative error




# Input transforms: the "hockey stick" ----
# houseprice is in the workspace
summary(houseprice)

# Create the formula for price as a function of squared size
(fmla_sqr <- price ~ I(size^2))

# Fit a model of price as a function of squared size (use fmla_sqr)
model_sqr <- lm(fmla_sqr, data = houseprice)

# Fit a model of price as a linear function of size
model_lin <- lm(price ~ size, data = houseprice)

# Make predictions and compare
houseprice %>% 
  mutate(pred_lin = predict(model_lin),       # predictions from linear model
         pred_sqr = predict(model_sqr)) %>%   # predictions from quadratic model 
  gather(key = modeltype, value = pred, pred_lin, pred_sqr) %>% # gather the predictions
  ggplot(aes(x = size)) + 
  geom_point(aes(y = price)) +                   # actual prices
  geom_line(aes(y = pred, color = modeltype)) + # the predictions
  scale_color_brewer(palette = "Dark2")




# Input transforms: the "hockey stick" (2) ----
# houseprice is in the workspace
summary(houseprice)

# fmla_sqr is in the workspace
(fmla_sqr <- price ~ I(size^2))

# Create a splitting plan for 3-fold cross validation
set.seed(34245)  # set the seed for reproducibility
splitPlan <- kWayCrossValidation(nrow(houseprice), 3, NULL, NULL)

# Sample code: get cross-val predictions for price ~ size
houseprice$pred_lin <- 0  # initialize the prediction vector
for (i in 1:3) {
  split <- splitPlan[[i]]
  model_lin <- lm(price ~ size, data = houseprice[split$train,])
  houseprice$pred_lin[split$app] <- predict(model_lin, newdata = houseprice[split$app,])
}

# Get cross-val predictions for price as a function of size^2 (use fmla_sqr)
houseprice$pred_sqr <- 0 # initialize the prediction vector
for (i in 1:3) {
  split <- splitPlan[[i]]
  model_sqr <- lm(fmla_sqr, data = houseprice[split$train, ])
  houseprice$pred_sqr[split$app] <- predict(model_sqr, newdata = houseprice[split$app, ])
}

# Gather the predictions and calculate the residuals
houseprice_long <- houseprice %>%
  gather(key = modeltype, value = pred, pred_lin, pred_sqr) %>%
  mutate(residuals = pred - price)

# Compare the cross-validated RMSE for the two models
houseprice_long %>% 
  group_by(modeltype) %>% # group by modeltype
  summarize(rmse = sqrt(mean(residuals^2)))




# Chapter 4: Dealing with Non-Linear Responses ----
# Fit a model of sparrow survival probability ----
# sparrow is in the workspace
summary(sparrow)

# Create the survived column
sparrow$survived <- ifelse(sparrow$status == "Survived", TRUE, FALSE)

# Create the formula
(fmla <- survived ~ total_length + weight + humerus)

# Fit the logistic regression model
sparrow_model <- glm(fmla, data = sparrow, family = binomial)

# Call summary
summary(sparrow_model)

# Call glance
(perf <- glance(sparrow_model))

# Calculate pseudo-R-squared
(pseudoR2 <- 1 - perf$deviance / perf$null.deviance)





# Predict sparrow survival ----
# sparrow is in the workspace
summary(sparrow)

# sparrow_model is in the workspace
summary(sparrow_model)

# Make predictions
sparrow$pred <- predict(sparrow_model, type = "response")

# Look at gain curve
GainCurvePlot(sparrow, "pred", "survived", "sparrow survival model")





# Fit a model to predict bike rental counts -----
# bikesJuly is in the workspace
str(bikesJuly)

# The outcome column
outcome <- "cnt"

# The inputs to use
vars <- c("hr", "holiday", "workingday", "weathersit", "temp",  "atemp",  "hum" , "windspeed")


# Create the formula string for bikes rented as a function of the inputs
(fmla <- paste(outcome, "~", paste(vars, collapse = " + ")))

# Calculate the mean and variance of the outcome
(mean_bikes <- mean(bikesJuly$cnt))
(var_bikes <- var(bikesJuly$cnt))

# Fit the model
bike_model <- glm(fmla, data = bikesJuly, family = quasipoisson)

# Call glance
(perf <- glance(bike_model))

# Calculate pseudo-R-squared
(pseudoR2 <- 1 - perf$deviance / perf$null.deviance)




# Predict bike rentals on new data ----
# bikesAugust is in the workspace
str(bikesAugust)

# bike_model is in the workspace
summary(bike_model)

# Make predictions on August data
bikesAugust$pred  <- predict(bike_model, newdata = bikesAugust, type = "response")

# Calculate the RMSE
bikesAugust %>% 
  mutate(residual = pred - cnt) %>%
  summarize(rmse  = sqrt(mean(residual^2)))

# Plot predictions vs cnt (pred on x-axis)
ggplot(bikesAugust, aes(x = pred, y = cnt)) +
  geom_point() + 
  geom_abline(color = "darkblue")




# Visualize the Bike Rental Predictions ----
# Plot predictions and cnt by date/time
quasipoisson_plot <- bikesAugust %>% 
  # set start to 0, convert unit to days
  mutate(instant = (instant - min(instant))/24) %>%  
  # gather cnt and pred into a value column
  gather(key = valuetype, value = value, cnt, pred) %>%
  filter(instant < 14) %>% # restric to first 14 days
  # plot value by instant
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Quasipoisson model")

quasipoisson_plot


# Model soybean growth with GAM ----
# soybean_train is in the workspace
summary(soybean_train)

# Plot weight vs Time (Time on x axis)
ggplot(soybean_train, aes(x = Time, y = weight)) + geom_point()

# Create the formula 
(fmla.lin <- weight ~ Time )

# Fit the Model.lin
model.lin <- lm(fmla.lin, data = soybean_train)

# Create the formula 
(fmla.gam <- weight ~ s(Time) )

# Fit the GAM Model
model.gam <- gam(fmla.gam, family = gaussian, data = soybean_train)

# Call summary() on model.lin and look for R-squared
summary(model.lin)

# Call summary() on model.gam and look for R-squared
summary(model.gam)

# Call plot() on model.gam
plot(model.gam)




# Predict with the soybean model on test data ----
# soybean_test is in the workspace
summary(soybean_test)

# Get predictions from linear model
soybean_test$pred.lin <- predict(model.lin, newdata = soybean_test)

# Get predictions from gam model
soybean_test$pred.gam <- as.numeric(predict(model.gam, newdata = soybean_test))

# Gather the predictions into a "long" dataset
soybean_long <- soybean_test %>%
  gather(key = modeltype, value = pred, pred.lin, pred.gam)

# Calculate the rmse
soybean_long %>%
  mutate(residual = weight - pred) %>%     # residuals
  group_by(modeltype) %>%                  # group by modeltype
  summarize(rmse = sqrt(mean(residual^2))) # calculate the RMSE

# Compare the predictions against actual weights on the test data
soybean_long %>%
  ggplot(aes(x = Time)) +                          # the column for the x axis
  geom_point(aes(y = weight)) +                    # the y-column for the scatterplot
  geom_point(aes(y = pred, color = modeltype)) +   # the y-column for the point-and-line plot
  geom_line(aes(y = pred, color = modeltype, linetype = modeltype)) + # the y-column for the point-and-line plot
  scale_color_brewer(palette = "Dark2")





# Tree-Based Methods -----
# Build a random forest model for bike rentals ----
# bikesJuly is in the workspace
str(bikesJuly)

# Random seed to reproduce results
seed <- 423563

# The outcome column
(outcome <- "cnt")

# The input variables
(vars <- c("hr", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"))

# Create the formula string for bikes rented as a function of the inputs
(fmla <- paste(outcome, "~", paste(vars, collapse = " + ")))

# Fit and print the random forest model
(bike_model_rf <- ranger(fmla, # formula 
                         bikesJuly, # data
                         num.trees = 500, 
                         respect.unordered.factors = "order", 
                         seed = seed))




# Predict bike rentals with the random forest model ----
# bikesAugust is in the workspace
str(bikesAugust)

# bike_model_rf is in the workspace
bike_model_rf

# Make predictions on the August data
bikesAugust$pred <- predict(bike_model_rf, bikesAugust)$predictions

# Calculate the RMSE of the predictions
bikesAugust %>% 
  mutate(residual = cnt - pred)  %>% # calculate the residual
  summarize(rmse  = sqrt(mean(residual^2)))      # calculate rmse

# Plot actual outcome vs predictions (predictions on x-axis)
ggplot(bikesAugust, aes(x = pred, y = cnt)) + 
  geom_point() + 
  geom_abline()




# Visualize random forest bike model predictions ----
# Print quasipoisson_plot
quasipoisson_plot

# Plot predictions and cnt by date/time
randomforest_plot <- bikesAugust %>% 
  mutate(instant = (instant - min(instant))/24) %>%  # set start to 0, convert unit to days
  gather(key = valuetype, value = value, cnt, pred) %>%
  filter(instant < 14) %>% # first two weeks
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Random Forest plot")


randomforest_plot







# vtreat on a small example ----
# !!!Kochrezept für Umwandlung Kategorialen Daten in die Dummy-Variablen ---- 
# dframe is in the workspace
dframe

# Create and print a vector of variable names
(vars <- c("color", "size"))

# Load the package vtreat
library(vtreat)

# Create the treatment plan
  treatplan <- designTreatmentsZ(dframe, vars)

# Examine the scoreFrame
(scoreFrame <- treatplan %>%
    use_series(scoreFrame) %>%
    select(varName, origName, code))

# We only want the rows with codes "clean" or "lev"
(newvars <- scoreFrame %>%
    filter(code %in% c("clean", "lev")) %>%
    use_series(varName))

# Create the treated training data
(dframe.treat <- prepare(treatplan, dframe, varRestriction = newvars))





# Novel levels ----
# treatplan is in the workspace
summary(treatplan)

# newvars is in the workspace
newvars

# Print dframe and testframe
dframe
testframe

# Use prepare() to one-hot-encode testframe
(testframe.treat <- prepare(treatplan, testframe, varRestriction = newvars))




# vtreat the bike rental data -----
# The outcome column
(outcome <- "cnt")

# The input columns
(vars <- c("hr", "holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"))

# Load the package vtreat
library(vtreat)

# Create the treatment plan from bikesJuly (the training data)
treatplan <- designTreatmentsZ(bikesJuly, vars, verbose = FALSE)

# Get the "clean" and "lev" variables from the scoreFrame
(newvars <- treatplan %>%
    use_series(scoreFrame) %>%        
    filter(code %in% c("clean", "lev")) %>%  # get the rows you care about
    use_series(varName))           # get the varName column

# Prepare the training data
bikesJuly.treat <- prepare(treatplan, bikesJuly,  varRestriction = newvars)

# Prepare the test data
bikesAugust.treat <- prepare(treatplan, bikesAugust,  varRestriction = newvars)

# Call str() on the treated data
str(bikesJuly.treat)
str(bikesAugust.treat)
# Note that the treated data does not include the outcome column. 



# Find the right number of trees for a gradient boosting machine ----
# The July data is in the workspace
# Run xgb.cv
cv <- xgb.cv(data = as.matrix(bikesJuly.treat), 
             label = bikesJuly$cnt,
             nrounds = 100,
             nfold = 5,
             objective = "reg:linear",
             eta = 0.3,
             max_depth = 6,
             early_stopping_rounds = 10,
             verbose = 0    # silent
)

# Get the evaluation log 
elog <- cv$evaluation_log

# Determine and print how many trees minimize training and test error
elog %>% 
  summarize(ntrees.train = which.min(train_rmse_mean),   # find the index of min(train_rmse_mean)
            ntrees.test  = which.min(test_rmse_mean))   # find the index of min(test_rmse_mean)

# In most cases, ntrees.test is less than ntrees.train. The training error keeps decreasing even after the test 
# error starts to increase. It's important to use cross-validation to find the right 
# number of trees (as determined by ntrees.test) and avoid an overfit model. 



# Fit an xgboost bike rental model and predict ----
# The number of trees to use, as determined by xgb.cv
ntrees <- 51

# Run xgboost
bike_model_xgb <- xgboost(data = as.matrix(bikesJuly.treat), # training data as matrix
                          label = bikesJuly$cnt,  # column of outcomes
                          nrounds = ntrees,       # number of trees to build
                          objective = "reg:linear", # objective
                          eta = 0.3,
                          depth = 6,
                          verbose = 0  # silent
)

# Make predictions
bikesAugust$pred <- predict(bike_model_xgb, as.matrix(bikesAugust.treat))

# Plot predictions (on x axis) vs actual bike rental count
ggplot(bikesAugust, aes(x = pred, y = cnt)) + 
  geom_point() + 
  geom_abline()




# Evaluate the xgboost bike rental model ----
# bikesAugust is in the workspace
str(bikesAugust)

# Calculate RMSE
bikesAugust %>%
  mutate(residuals = cnt - pred) %>%
  summarize(rmse = sqrt(mean(residuals^2)))




# Visualize the xgboost bike rental model ----
# Print quasipoisson_plot
quasipoisson_plot

# Print randomforest_plot
randomforest_plot

# Plot predictions and actual bike rentals as a function of time (days)
gradientboosting_plot <- bikesAugust %>% 
  mutate(instant = (instant - min(instant))/24) %>%  # set start to 0, convert unit to days
  gather(key = valuetype, value = value, cnt, pred) %>%
  filter(instant < 14) %>% # first two weeks
  ggplot(aes(x = instant, y = value, color = valuetype, linetype = valuetype)) + 
  geom_point() + 
  geom_line() + 
  scale_x_continuous("Day", breaks = 0:14, labels = 0:14) + 
  scale_color_brewer(palette = "Dark2") + 
  ggtitle("Predicted August bike rentals, Gradient Boosting model")

gradientboosting_plot