library(readxl)
library(ggplot2)
library(forecast)
library(fpp)
library(fpp2)

# 1. Chapter: Exploring and visualizing time series in R ----
  # Creating time series objects in R ----
  # Read the data from Excel into R
  mydata <- read_excel("exercise1.xlsx")
  
  # Create a ts object called myts
  myts <- ts(mydata[,-1], start = c(1981, 1), frequency = 4)
  
  
  # Time series plots ----
  # Plot the data with facetting
  autoplot(myts, facets = TRUE)
  
  # Plot the data without facetting
  autoplot(myts, facets = FALSE)
  
  # Plot the four series
  autoplot(gold, facets = FALSE)
  autoplot(woolyrnq, facets = FALSE)
  autoplot(gas, facets = FALSE)
  autoplot(taylor, facets = FALSE)
  
  # Find the outlier in the gold series
  goldoutlier <- which.max(gold)
  
  # Create a vector of the seasonal frequencies of gold, woolyrnq, gas, and taylor
  frequency(gold)
  frequency(woolyrnq)
  frequency(gas)
  frequency(taylor)
  
  freq <- c(1, 4, 12, 336)
  
  
  
  # Seasonal plots ----
  # a plots of the a10 data
  autoplot(a10, facets = TRUE)
  ggseasonplot(a10)
  
  # Produce a polar coordinate season plot for the a10 data
  ggseasonplot(a10, polar = TRUE)
  
  # Restrict the ausbeer data to start in 1992
  beer <- window(ausbeer, start = 1992)
  
  # Make plots of the beer data
  autoplot(beer, facets = TRUE)
  ggsubseriesplot(beer)
  
  
  
  # Autocorrelation of non-seasonal time series ----
  # Create an autoplot of the oil data
  autoplot(oil)
  
  # Create a lag plot of the oil data
  gglagplot(oil, 9)
  
  # Create an ACF plot of the oil data
  ggAcf(oil)
  
  
  
  # Autocorrelation of seasonal and cyclic time series ----
  # Plot the annual sunspot numbers
  autoplot(sunspot.year)
  ggAcf(sunspot.year)
  
  # Save the lag corresponding to maximum autocorrelation
  maxlag_sunspot <- 1
  
  # Plot the traffic on the Hyndsight blog
  autoplot(hyndsight)
  ggAcf(hyndsight)
  
  # Save the lag corresponding to maximum autocorrelation
  maxlag_hyndsight <- 7
  
  
  
  
  # Stock prices and white noise ----
  # Plot the original series
  autoplot(goog)
  
  # Plot the differenced series
  autoplot(diff(goog))
  
  # ACF of the differenced series
  ggAcf(diff(goog))
  
  # Ljung-Box test of the differenced series
  Box.test(diff(goog), lag = 10, type = "Ljung")
  
  
  
  
  
# 2. Chapter: Benchmark methods and forecast accuracy ----
  # Naive forecasting methods ----
  # Use naive() to forecast the goog series
  fcgoog <- naive(goog, 20)
  
  # Plot and summarize the forecasts
  autoplot(fcgoog)
  summary(fcgoog)
  
  # Use snaive() to forecast the ausbeer series
  fcbeer <- snaive(ausbeer, 16)
  
  # Plot and summarize the forecasts
  autoplot(fcbeer)
  summary(fcbeer)
  
  
  
  # Checking time series residuals ----
  # Check the residuals from the naive forecasts applied to the goog series
  goog %>% naive() %>% checkresiduals()
  
  # Do they look like white noise (TRUE or FALSE)
  googwn <- TRUE
  
  # Check the residuals from the seasonal naive forecasts applied to the ausbeer series
  ausbeer %>% snaive() %>% checkresiduals()
  
  # Do they look like white noise (TRUE or FALSE)
  beerwn <- FALSE
  
  
  
  
  # Evaluating forecast accuracy of non-seasonal methods ----
  # Create the training data as train
  train <- subset.ts(gold, end = 1000)
  
  # Compute naive forecasts and save to naive_fc
  naive_fc <- naive(train, h = 108)
  
  # Compute mean forecasts and save to mean_fc
  mean_fc <- meanf(train, h = 108)
  
  # Use accuracy() to compute RMSE statistics
  accuracy(naive_fc, gold)
  accuracy(mean_fc, gold)
  
  # Assign one of the two forecasts as bestforecasts
  bestforecasts <- naive_fc
  
  
  
  # Evaluating forecast accuracy of seasonal methods ----
  # Create three training series omitting the last 1, 2, and 3 years
  train1 <- window(vn[, "Melbourne"], end = c(2010, 4))
  train2 <- window(vn[, "Melbourne"], end = c(2009, 4))
  train3 <- window(vn[, "Melbourne"], end = c(2008, 4))
  
  # Produce forecasts using snaive()
  fc1 <- snaive(train1, h = 4)
  fc2 <- snaive(train2, h = 4)
  fc3 <- snaive(train3, h = 4)
  
  # Use accuracy() to compare the MAPE of each series
  accuracy(fc1, vn[, "Melbourne"])["Test set", "MAPE"]
  accuracy(fc2, vn[, "Melbourne"])["Test set", "MAPE"]
  accuracy(fc3, vn[, "Melbourne"])["Test set", "MAPE"]
  
  
  
  
  
  # Using tsCV() for time series cross-validation ----
  # Compute cross-validated errors for up to 8 steps ahead
  e <- matrix(NA_real_, nrow = 1000, ncol = 8)
  for (h in 1:8)
    e[, h] <- tsCV(goog, naive, h = h)
  
  # Compute the MSE values and remove missing values
  mse <- colMeans(e^2, na.rm = TRUE)
  
  # Plot the MSE values (y) against the forecast horizon (x)
  data.frame(h = 1:8, MSE = mse) %>%
    ggplot(aes(x = h, y = MSE)) + geom_point()
  
  
  
  
# 3. Chapter: Exponential smoothing ----
  # Simple exponential smoothing ----
  # Use ses() to forecast the next 10 years of winning times
  fc <- ses(marathon, h = 10)
  
  # Use summary() to see the model parameters
  summary(fc)
  
  # Use autoplot() to plot the forecasts
  autoplot(fc)
  
  # Add the one-step forecasts for the training data to the plot
  autoplot(fc) + autolayer(fitted(fc))
  
  
  
  # SES vs naive ----
  # Create a training set using subset()
  train <- subset(marathon, end = length(marathon) - 20)
  
  # Compute SES and naive forecasts, and save to fcses and fcnaive
  fcses <- ses(train, h = 20)
  fcnaive <- naive(train, h = 20)
  
  # Calculate forecast accuracy measures
  accuracy(fcses, marathon)
  accuracy(fcnaive, marathon)
  
  # Save the better forecasts as fcbest
  fcbest <- fcnaive
  
  
  
  
  # Holt's trend methods ----
  # Produce 10 year forecasts of austa using holt()
  fcholt <- holt(austa, h = 10)
  
  # Look at fitted model using summary()
  summary(fcholt)
  
  # Plot the forecasts
  autoplot(fcholt)
  
  # Check that the residuals look like white noise
  checkresiduals(fcholt)
  
  
  
  # Holt-Winters with monthly data ----
  # Plot the data
  autoplot(a10)
  
  # Produce 3 year forecasts
  fc <- hw(a10, h = 36, seasonal = "multiplicative")
  
  # Check residuals look like white noise (set whitenoise to be TRUE or FALSE)
  checkresiduals(fc)
  whitenoise <- FALSE
  
  # Plot forecasts
  autoplot(fc)
  
  
  
  # Holt-Winters method with daily data ----
  # Create training data with subset()
  train <- subset(hyndsight, end = length(hyndsight) - 28)
  
  # Holt-Winters additive forecasts as fchw
  fchw <- hw(train, seasonal = "additive", h = 28)
  
  # Seasonal naive forecasts as fcsn
  fcsn <- snaive(train, h = 28)
  
  # Find better forecasts with accuracy()
  accuracy(fchw, hyndsight)
  accuracy(fcsn, hyndsight)
  
  # Plot the better forecasts
  autoplot(fchw)
  
  
  
  # Automatic forecasting with exponential smoothing ----
  # Fit ETS model to austa in fitaus
  fitaus <- ets(austa)
  
  # Check residuals
  checkresiduals(fitaus)
  
  # Plot forecasts
  autoplot(forecast(fitaus))
  
  # Repeat for hyndsight data in fiths
  fiths <- ets(hyndsight)
  checkresiduals(fiths)
  autoplot(forecast(fiths))
  
  # Which model(s) fails test? (TRUE or FALSE)
  fitausfail <- FALSE
  fithsfail <- TRUE
  
  
  
  # ets vs snaive ----
  # Function to return ETS forecasts
  fets <- function(y, h) {
    forecast(ets(y), h = h)
  }
  
  # Apply tsCV() for both methods
  e1 <- tsCV(cement, fets, h = 4)
  e2 <- tsCV(cement, snaive, h = 4)
  
  # Compute MSE of resulting errors (watch out for missing values)
  mean(e1^2, na.rm = TRUE)
  mean(e2^2, na.rm = TRUE)
  
  # Best the better MSE as bestmse
  bestmse <- mean(e2^2, na.rm = TRUE)
  
  
  
  
  # When does ets fail? ----
  # It's important to realize that ETS doesn't work for all cases. 
  # Plot the lynx series
  autoplot(lynx)
  
  # Use ets() to model the lynx series
  fit <- ets(lynx)
  
  # Use summary() to look at model and parameters
  summary(fit)
  
  # Plot 20-year forecasts of the lynx series
  lynx %>% ets() %>% forecast(h = 20) %>% autoplot()
  
  
  
  
  
  
  
# 3. Chapter: Forecasting with ARIMA models  ----
  # Box-Cox transformations for time series ----
  # Plot the series
  autoplot(a10)
  
  # Try four values of lambda in Box-Cox transformations
  a10 %>% BoxCox(lambda = 0) %>% autoplot()
  a10 %>% BoxCox(lambda = 1) %>% autoplot()
  a10 %>% BoxCox(lambda = -1) %>% autoplot()
  a10 %>% BoxCox(lambda = 2) %>% autoplot()
  
  # Compare with BoxCox.lambda()
  BoxCox.lambda(a10)
  
  
  
  # Non-seasonal differencing for stationarity ----
  # Plot the US female murder rate
  autoplot(wmurders)
  
  # Plot the differenced murder rate
  autoplot(diff(wmurders))
  
  # Plot the ACF of the differenced murder rate
  ggAcf(diff(wmurders))
  
  
  
  # Seasonal differencing for stationarity ----
  # Plot the data
  autoplot(h02)
  
  # Take logs and seasonal differences of h02
  difflogh02 <- diff(log(h02), lag = 12)
  
  # Plot the resulting logged and differenced data
  autoplot(difflogh02)
  
  # Take another difference and plot
  ddifflogh02 <- diff(difflogh02)
  autoplot(ddifflogh02)
  
  # Plot ACF of final series ddifflogh02
  ggAcf(ddifflogh02)
  
  
  
  
  # Automatic ARIMA models for non-seasonal time series ----
  # Fit an automatic ARIMA model to the austa series
  fit <- auto.arima(austa)
  
  # Check the residuals look like white noise (set residualsok to TRUE or FALSE)
  checkresiduals(fit)
  residualsok <- TRUE
  
  # Summarize the model
  summary(fit)
  
  # Find the AICc value and the number of differences used
  AICc <- -14.46
  d <- 1
  
  # Plot forecasts of fit
  fit %>% forecast(h = 10) %>% autoplot()
  
  
  
  # Forecasting with ARIMA models ----
  # Plot forecasts from an ARIMA(0,1,1) model with no drift
  austa %>% Arima(order = c(0, 1, 1), include.constant = FALSE) %>% forecast() %>% autoplot()
  
  # Plot forecasts from an ARIMA(2,1,3) model with drift
  austa %>% Arima(order = c(2, 1, 3), include.constant = TRUE) %>% forecast() %>% autoplot()
  
  # Plot forecasts from an ARIMA(0,0,1) model with a constant.
  austa %>% Arima(order = c(0, 0, 1), include.constant = TRUE) %>% forecast() %>% autoplot()
  
  # Plot forecasts from an ARIMA(0,2,1) model with no constant.
  austa %>% Arima(order = c(0, 2, 1), include.constant = FALSE) %>% forecast() %>% autoplot()
  
  
  
  
  # Comparing auto.arima() and ets() on non-seasonal data ----
  # Set up forecast functions for ETS and ARIMA models
  fets <- function(x, h) {
    forecast(ets(x), h = h)
  }
  farima <- function(x, h) {
    forecast(auto.arima(x), h = h)
  }
  
  # Compute CV errors for ETS as e1
  e1 <- tsCV(austa, fets, h = 1)
  
  # Compute CV errors for ARIMA as e2
  e2 <- tsCV(austa, farima, h = 1)
  
  # Find MSE of each model class
  mean(e1^2, na.rm = TRUE)
  mean(e2^2, na.rm = TRUE)
  
  # Plot 10-year forecasts using the best model class
  austa %>% auto.arima() %>% forecast(h = 10) %>% autoplot()
  
  
  
  
  # Automatic ARIMA models for seasonal time series ----
  # Check that the logged h02 data have stable variance
  h02 %>% log() %>% autoplot()
  
  # Fit a seasonal ARIMA model to h02 with lambda = 0
  fit <- auto.arima(h02, lambda = 0)
  
  # Summarize the fitted model
  summary(fit)
  
  # Record the amount of lag-1 differencing and seasonal differencing used
  d <- 1
  D <- 1
  
  # Plot 2-year forecasts
  fit %>% forecast(h = 24) %>% autoplot()
  
  
  
  # Exploring auto.arima options ----
  # Use the default options to find an ARIMA model for euretail
  fit1 <- auto.arima(euretail)
  
  # Don't use a stepwise search.
  fit2 <- auto.arima(euretail, stepwise = FALSE)
  
  # AICc of best model
  summary(fit2)
  AICc <- 68.39
  
  # Compute 2-year forecasts from best model
  euretail %>% auto.arima(stepwise = FALSE) %>% forecast(h = 8) %>% autoplot()
  
  
  
  # Comparing auto.arima() and ets() on seasonal data ----
  # Use 20 years of the qcement data beginning in 1988
  train <- window(qcement, start = 1988, end = c(2007, 4))
  
  # Fit an ARIMA and an ETS model to the training data
  fit1 <- ets(train)
  fit2 <- auto.arima(train)
  
  # Check that both models have white noise residuals
  checkresiduals(fit1)
  checkresiduals(fit2)
  
  # Produce forecasts for each model
  fc1 <- forecast(fit1, h = 25)
  fc2 <- forecast(fit2, h = 25)
  
  # Use accuracy() to find best model based on MSE
  accuracy(fc1, qcement)
  accuracy(fc2, qcement)
  best <- fc1
  
  
  
  
  
# 5. Chapter: Advanced methods  -----
  # Forecasting sales allowing for advertising expenditure ----
  # Time plot of both variables
  autoplot(advert, facets = TRUE)
  
  # Fit ARIMA model
  fit <- auto.arima(advert[, "sales"], xreg = advert[, "advert"], stationary = TRUE)
  
  # Check model. Increase in sales for each unit increase in advertising
  salesincrease <-coefficients(fit)[3]
  
  # Forecast fit as fc
  fc <- forecast(fit, xreg = rep(10,6))
  
  # Plot forecasts
  autoplot(fc) + xlab("Year") + ylab("Sales")
  
  
  
  # Forecasting electricity demand ----
  # Time plots of demand and temperatures
  autoplot(elec[, c("Demand", "Temperature")], facets = TRUE)
  
  # Matrix of regressors
  xreg <- cbind(MaxTemp = elec[, "Temperature"], 
                MaxTempSq = elec[, "Temperature"]^2, 
                Workday = elec[, "Workday"])
  
  # Fit model
  fit <- auto.arima(elec[, "Demand"], xreg = xreg)
  
  # Forecast one day ahead
  forecast(fit, xreg = cbind(20, 20^2, 1))
  
  
  
  # Forecasting weekly data ----
  # Set up harmonic regressors of order 13
  harmonics <- fourier(gasoline, K = 13)
  
  # Fit regression model with ARIMA errors
  # FALSE, weil die Saisonalitaet schon durch Fourier-Transformation modelliert wird
  fit <- auto.arima(gasoline, xreg = harmonics, seasonal = FALSE)
  
  # Forecasts next 3 years
  newharmonics <- fourier(gasoline, K = 13, h = 3 * 52)
  fc <- forecast(fit, xreg = newharmonics)
  
  # Plot forecasts
  autoplot(fc)
  
  
  
  
  # Harmonic regression for multiple seasonality ----
  # Fit a harmonic regression using order 10 for each type of seasonality.
  fit <- tslm(taylor ~ fourier(taylor, K = c(10, 10)))
  
  # Forecast 20 working days ahead
  fc <- forecast(fit, newdata = data.frame(fourier(taylor, K = c(10, 10), h = 20 * 48)))
  
  # Plot the forecasts
  autoplot(fc)
  
  # Check the residuals
  checkresiduals(fit)
  
  
  
  
  # Forecasting call bookings ----
  # Plot the calls data
  autoplot(calls)
  
  # Set up the xreg matrix
  xreg <- fourier(calls, K = c(10,0))
  
  # Fit a dynamic regression model
  fit <- auto.arima(calls, xreg = xreg, seasonal = FALSE, stationary = TRUE)
  
  # Check the residuals
  checkresiduals(fit)
  
  # Plot forecasts for 10 working days ahead
  fc <- forecast(fit, xreg =  fourier(calls, c(10, 0), 1690))
  autoplot(fc)
  
  
  
  
  
  # TBATS models for electricity demand ----
  # Plot the gas data
  autoplot(gas)
  
  # Fit a TBATS model to the gas data
  fit <- tbats(gas)
  
  # Forecast the series for the next 5 years
  fc <- forecast(fit, h = 5 * 12)
  
  # Plot the forecasts
  autoplot(fc)
  
  # Record the Box-Cox parameter and the order of the Fourier terms
  lambda <- 0.082
  K <- 5