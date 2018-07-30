# 1. Chapter 1: Apply  -----
# 1.1. Plotting financial data ----
    # Get SPY from Yahoo Finance ("yahoo")
    getSymbols("SPY", from = "2000-01-01", to = "2016-06-30", src =  "yahoo", adjust =  TRUE)
    
    # Plot the closing price of SPY
    plot(Cl(SPY))

    
# 1.2. Adding a moving average to financial data ----
    # Plot the closing prices of SPY
    plot(Cl(SPY))
    
    # Add a 200-day moving average using the lines command
    lines(SMA(Cl(SPY), n = 200), col = "red")
    
    
# 2. Chapter 2: A boilerplate for quantstrat strategies   -----
# 2.1. Understanding initialization settings - I ----
    # Load the quantstrat-package
    library(quantstrat)
    
    # Create initdate, from, and to charater strings
    initdate <- "1999-01-01"
    from <- "2003-01-01"
    to <- "2015-12-31"
    
    # Set the timezone to UTC
    Sys.setenv(TZ = "UTC")
    
    # Set the currency to USD 
    currency("USD")
    
    
# 2.2. Understanding initialization settings - II ----
    # Load the quantmod-package
    library(quantmod)
    
    # Retrieve SPY from yahoo
    getSymbols("SPY", from = "2003-01-01", to = "2015-12-31", src = "yahoo", adjust = TRUE)
    
    # Use the stock command to initialize SPY and set currency to USD
    stock("SPY", currency = "USD")
    
    
# 2.3. Understanding initialization settings - III ----
    # Define your trade size and initial equity
    tradesize <- 100000
    initeq <- 100000
    
    # Define the names of your strategy, portfolio and account
    strategy.st <- "firststrat"
    portfolio.st <- "firststrat"
    account.st <- "firststrat"
    
    # Remove the existing strategy if it exists
    rm.strat(strategy.st)
    
# 2.4. Understanding initialization settings - III ----
    # initialize the portfolio
    initPortf(portfolio.st, symbols = "SPY", initDate = initdate, currency = "USD")
    
    # initialize the account
    initAcct(account.st, portfolios = portfolio.st, initDate = initdate, currency = "USD", initEq = initeq)
    
    # initialize the orders
    initOrders(portfolio.st, initDate = initdate)
    
    # store the strategy
    strategy(strategy.st, store = TRUE)
    
    
# 3. Chapter 3: Indicators   -----
# 3.1. The SMA and RSI functions ----
    # Create a 200-day moving average
    spy_sma <- SMA(x = Cl(SPY), n = 200)
    
    # Create an RSI with a 3 day lookback period
    spy_rsi <- RSI(price = Cl(SPY), n = 3)
    
# 3.2. The SMA and RSI functions ----
    # Plot the closing prices of SPY
    plot(Cl(SPY))
    
    # Overlay a 200-day SMA
    lines(SMA(Cl(SPY), n = 200), col = "red")
    
    # Is this a trend or reversion indicator?
    "trend"
    
# 3.3. The SMA and RSI functions ----
    # plot the closing price of SPY
    plot(Cl(SPY))
    
    # plot the RSI 2
    plot(RSI(Cl(SPY), n = 2))
    
    # trend or reversion?
    "reversion"
    
# 3.4. The SMA and RSI functions ----
    # Add a 200-day simple moving average indicator to your strategy
    add.indicator(strategy = strategy.st, 
                  
                  # Add the SMA function
                  name = "SMA", 
                  
                  # Create a lookback period
                  arguments = list(x = quote(Cl(mktdata )), n = 200), 
                  
                  # Label your indicator SMA200
                  label = "SMA200")

# 3.5. Implementing an indicator - II ----
    # Add a 50-day simple moving average indicator to your strategy
    add.indicator(strategy = strategy.st, 
                  
                  # Add the SMA function
                  name = "SMA", 
                  
                  # Create a lookback period
                  arguments = list(x = quote(Cl(mktdata)), n = 50), 
                  
                  # Label your indicator SMA50
                  label = "SMA50")
    
# 3.6. Implementing an indicator - III ----
    # add an RSI 3 indicator to your strategy
    add.indicator(strategy = strategy.st, 
                  
                  # add an RSI function to your strategy
                  name = "RSI", 
                  
                  # use a lookback period of 3 days
                  arguments = list(price = quote(Cl(mktdata)), n = 3), 
                  
                  # label it RSI_3
                  label = "RSI_3")
    
# 3.7. Code your own indicator - I ----
    # Write the RSI_avg function
    RSI_avg <- function(price, n1, n2) {
      
      # RSI 1 takes an input of the price and n1
      rsi_1 <- RSI(price = price, n = n1)
      
      # RSI 2 takes an input of the price and n2
      rsi_2 <- RSI(price = price, n = n2)
      
      # RSI_avg is the average of rsi_1 and rsi_2
      RSI_avg <- (rsi_1 + rsi_2)/2
      
      # Your output of RSI_avg needs a column name of "RSI_avg"
      colnames(RSI_avg) <- "RSI_avg"
      return(RSI_avg)
    }
    
    # Add the RSI_avg function to your strategy using an n1 of 3 and an n2 of 4, and label it "RSI_3_4"
    add.indicator(strategy.st, name = "RSI_avg", arguments = list(price = quote(Cl(mktdata)), n1 = 3, n2 = 4), label = "RSI_3_4")
    
# 3.8. Code your own indicator - II ----
    # Declare the DVO function. The first argument is the high, low, and close of market data.
    DVO <- function(HLC, navg = 2, percentlookback = 126) {
      
      # Compute the ratio between closing prices to the average of high and low
      ratio <- Cl(HLC)/((Hi(HLC) + Lo(HLC))/2)
      
      # Smooth out the ratio outputs using a moving average
      avgratio <- SMA(ratio, n = navg)
      
      # Convert ratio into a 0-100 value using runPercentRank function
      out <- runPercentRank(avgratio, n = percentlookback, exact.multiplier = 1) * 100
      colnames(out) <- "DVO"
      return(out)
    }
    
# 3.9. Apply your own indicator ----
    # add the DVO indicator to your strategy
    add.indicator(strategy = strategy.st, name = "DVO", 
                  arguments = list(HLC = quote(HLC(mktdata)), navg = 2, percentlookback = 126),
                  label = "DVO_2_126")
    
    # use applyIndicators to test out your indicators
    test <- applyIndicators(strategy = strategy.st, mktdata = OHLC(SPY))
    
    # subset your data between Sep. 1 and Sep. 5 of 2013
    test_subset <- test["2013-09-01/2013-09-05"]
    
    
# 4. Chapter 4: Signals   -----
# 4.1. Using sigComparison ----
    # add a sigComparison which specifies that SMA50 must be greater than SMA200, call it longfilter
    add.signal(strategy.st, name = "sigComparison", 
               
               # we are interested in the relationship between the SMA50 and the SMA200
               arguments = list(columns = c("SMA50", "SMA200"), 
                                
                                # particularly, we are interested when the SMA50 is greater than the SMA200
                                relationship = "gt"),
               
               # label this signal longfilter
               label = "longfilter")
    
# 4.2. Using sigCrossover ----
    # add a sigCrossover which specifies that the SMA50 is less than the SMA200 and label it filterexit
    add.signal(strategy.st, name = "sigCrossover",
               
               # we're interested in the relationship between the SMA50 and the SMA200
               arguments = list(columns = c("SMA50", "SMA200"),
                                
                                # the relationship is that the SMA50 crosses under the SMA200
                                relationship = "lt"),
               
               # label it filterexit
               label = "filterexit")
    
# 4.3. Using sigCrossover ----
    