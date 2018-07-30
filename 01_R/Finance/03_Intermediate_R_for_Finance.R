# 4. Chapter 4: Functions  -----
# 4.X. tidyquant package ----
    # Library tidquant
    library(tidyquant)
    
    # Pull Apple stock data
    apple <- tq_get("AAPL", get = "stock.prices")
    
    # Take a look at what it returned
    head(apple)
    
    # Plot the stock price over time
    plot(apple$date, apple$adjusted, type = "l")
    
    # Calculate daily stock returns for the adjusted price
    apple <- tq_mutate(data = apple,
                       ohlc_fun = Ad,
                       mutate_fun = dailyReturn)
    
    # Sort the returns from least to greatest
    sorted_returns <- sort(apple$daily.returns)
    
    # Plot them
    plot(sorted_returns)
    

# 5. Chapter 5: Apply  -----
# 5.1. sapply() VS lapply() ----
    # lapply() on stock_return
    lapply(stock_return, sharpe)
    
    # sapply() on stock_return
    sapply(stock_return, sharpe)
    
    
    # sapply() on stock_return with optional arguments
    sapply(stock_return, sharpe, simplify = FALSE, USE.NAMES = FALSE)
    
    
# 5.2. vapply() VS sapply() ----
    # bei vapply() wird die das erwartete Output durch FUN.VALUE im voraus spezifiziert !!!!!!
    # Sharpe ratio for all stocks
    vapply(stock_return, sharpe, FUN.VALUE = numeric(1))
    
    # Summarize Apple
    summary(stock_return$apple)
    
    # Summarize all stocks
    vapply(stock_return, summary, FUN.VALUE = numeric(6))
    
    # Max and min
    vapply(stock_return, 
           FUN = function(x) { c(max(x), min(x)) }, 
           FUN.VALUE = numeric(2))
    
    