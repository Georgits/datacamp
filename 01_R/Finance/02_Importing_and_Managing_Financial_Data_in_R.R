require(quantmod)
require(Quandl)

# split-lapply-rbind paradigm
# x_split <- split(x, f = "months")
# x_list <- lapply(x_split, cummax)
# x_list_rbind <- do.call(rbind, x_list)


# 1. Chapter 1: Introduction and downloading data  -----
# 1.1. Introducing getSymbols() ----
    # Load the quantmod package
    library(quantmod)
    
    # Import QQQ data from Yahoo! Finance
    getSymbols("QQQ", auto_assign = TRUE)
    
    # Look at the structure of the object getSymbols created
    str(QQQ)
    
    # Look at the first few rows of QQQ
    head(QQQ)

    
# 1.2. Data sources ----
    # Import QQQ data from Google Finance
    getSymbols("QQQ", src = "google", auto_assign = TRUE)
    
    # Look at the structure of QQQ
    str(QQQ)
    
    # Import GDP data from FRED
    getSymbols("GDP", src = "FRED", auto_assign = TRUE)
    
    # Look at the structure of GDP
    str(GDP)
    
    
# 1.3. Make getSymbols() return the data it retrieves ----
    # Load the quantmod package
    library(quantmod)
    
    # Assign SPY data to 'spy' object using auto.assign argument
    spy <- getSymbols("SPY", auto.assign = FALSE)
    
    # Look at the structure of the 'spy' object
    str(spy)
    
    # Assign JNJ data to 'jnj' object using env argument
    jnj <- getSymbols("JNJ", env = NULL, src = "yahoo")
    
    # Look at the structure of the 'jnj' object
    str(jnj)
    
    
    
# 1.4. Introducing Quandl() ----
    # Load the Quandl package
    library(Quandl)
    
    # Import GDP data from FRED
    gdp <- Quandl("FRED/GDP")
    
    # Look at the structure of the object returned by Quandl
    str(gdp)
    
    
# 1.5. Return data type ----
    # Import GDP data from FRED as xts
    gdp_xts <- Quandl("FRED/GDP", type = "xts")
    
    # Look at the structure of gdp_xts
    str(gdp_xts)
    
    # Import GDP data from FRED as zoo
    gdp_zoo <- Quandl("FRED/GDP", type = "zoo")
    
    
    # Look at the structure of gdp_zoo
    str(gdp_zoo)
    
    
# 1.6. Find stock ticker from Google Finance ----
    # Create an object containing the Pfizer ticker symbol
    symbol <- "PFE"
    
    # Use getSymbols to import the data
    getSymbols(symbol, src = "google", auto.assign = TRUE)
    
    # Look at the first few rows of data
    head(PFE)
    
    
# 1.7. Download exchange rate data from Oanda  ----
    # Create a currency_pair object
    # quantmod::oanda.currencies contains a list of currencies provided by Oanda.com
    currency_pair <- "GBP/CAD"
    
    # Load British Pound to Canadian Dollar exchange rate data
    getSymbols(currency_pair,  src = "oanda", auto.assign = TRUE)
    
    # Examine object using str()
    str(GBPCAD)
    
    # Try to load more than 500 days
    getSymbols(currency_pair, from = "2001-01-01", to = "2016-01-01", src = "oanda", auto.assign = TRUE)
    
    
    
# 1.8. Find and download US Civilian Unemployment Rate data from FRED  ----
    # Create a series_name object
    series_name <- "UNRATE"
    
    # Load the data using getSymbols
    getSymbols(series_name, src = "FRED", auto.assign = TRUE)
    
    # Create a quandl_code object
    quandl_code <- "FRED/UNRATE"
    
    # Load the data using Quandl
    unemploy_rate <- Quandl(code = quandl_code)
    
    
    
# 2. Chapter 2: Extracting and transforming data  -----
# 2.1. Extract one column from one instrument ----
    # Look at the head of DC
    head(DC)
    
    # Extract the close column
    dc_close <- Cl(DC)
    
    # Look at the head of dc_close
    head(dc_close)
    
    # Extract the volume column
    dc_volume <- Vo(DC)
    
    # Look at the head of dc_volume
    head(dc_volume)
    
    
# 2.2. Extract multiple columns from one instrument ----
    # Extract the high, low, and close columns
    dc_hlc <- HLC(DC)
    
    # Look at the head of dc_hlc
    head(dc_hlc)
    
    # Extract the open, high, low, close, and volume columns
    dc_ohlcv <- OHLCV(DC)
    
    # Look at the head of dc_ohlcv
    head(dc_ohlcv)
    
# 2.3. Use getPrice to extract other columns ----
    # Download CME data for CL and BZ as an xts object
    oil_data <- Quandl(code = c("CME/CLH2016", "CME/BZH2016"), type = "xts")
    
    # Look at the column names of the oil_data object
    colnames(oil_data)
    
    # Extract the Open price for CLH2016
    cl_open <- getPrice(oil_data, symbol = "CLH2016", prefer = "Open$")
    
    # Look at January, 2016 using xts' ISO-8601 subsetting
    cl_open["2016-01",1]
    

# 2.4. Use Quandl to download weekly returns data ----
    # CL and BZ Quandl codes
    quandl_codes <- c("CME/CLH2016","CME/BZH2016")
    
    # Download quarterly CL and BZ prices
    qtr_price <- Quandl(quandl_codes, collapse ="quarterly", type = "xts")
    
    # View the high prices for both series
    Hi(qtr_price)
    
    # Download quarterly CL and BZ returns
    qtr_return <- Quandl(quandl_codes, collapse ="quarterly", type = "xts", transform = "rdiff")
    
    # View the settle price returns for both series
    getPrice(qtr_return, prefer = "Settle")
    
    
# 2.5. Combine objects from an environment using do.call and eapply ----
    data_env <- new.env()
    getSymbols(c("SPY", "QQQ"), env = data_env, auto.assign = TRUE)
    # Call head on each object in data_env using eapply
    data_list <- eapply(data_env, head)
    
    # Merge all the list elements into one xts object
    data_merged <- do.call(merge, data_list)
    
    # Ensure the columns are ordered: open, high, low, close
    data_ohlc <- OHLC(data_merged)
    
    
# 2.6. Use quantmod to download multiple instruments and extract the close column ----
    # Symbols
    symbols <- c("AAPL", "MSFT", "IBM")
    
    # Create new environment
    data_env = new.env()
    
    # Load symbols into data_env
    getSymbols(symbols, env = data_env)
    
    # Extract the close column from each object and merge into one xts object
    close_data <- do.call(merge, eapply(data_env, Cl))
    
    # View the head of close_data
    head(close_data)
    
    
# 3. Chapter 3: Setting the default data source  -----
# 3.1. Setting the default data source ----
    # Set the default to pull data from Google Finance
    setDefaults(getSymbols, src = "google")
    
    # Get GOOG data
    getSymbols("GOOG", auto.assign = TRUE)
    
    # Verify the data was actually pulled from Google Finance
    str(GOOG)
    
    
# 3.2. Set default arguments for a getSymbols source method ----
    # Look at getSymbols.yahoo arguments
    args(getSymbols.yahoo)
    
    # Set default 'from' value for getSymbols.yahoo
    setDefaults(getSymbols.yahoo, from = "2000-01-01")
    
    # Confirm defaults were set correctly
    getDefaults("getSymbols.yahoo")
    
    
# 3.3. Set default data source for one symbol ----
    # getSymbols("CP", src = "yahoo")
    # Look at the first few rows of CP
    head(CP)
    
    # Set the source for CP to FRED
    setSymbolLookup(CP = list(src = "FRED"))
    
    # Load CP data again
    getSymbols("CP")
    
    # Look at the first few rows of CP
    head(CP)
    
    
# 3.4. Save and load symbol lookup table ----
    # Save symbol lookup table
    saveSymbolLookup(file = "my_symbol_lookup.rda")
    
    # Set default source for CP to "yahoo"
    setSymbolLookup(CP = list(src = "yahoo"))
    
    # Verify the default source is "yahoo"
    getSymbolLookup("CP")
    
    # Load symbol lookup table
    loadSymbolLookup("my_symbol_lookup.rda")
    
    # Verify the default source is "FRED"
    getSymbolLookup("CP")
    
    
# 3.5. Access the object using get() or backticks ----
    # Load BRK-A data
    getSymbols("BRK-A", auto.assign = TRUE)
    
    # Use backticks and head() to look at the loaded data
    head(`BRK-A`)
    
    # Use get() to assign the BRK-A data to an object named BRK.A
    BRK.A <- get("BRK-A")
    
    
# 3.6. Create valid name for one instrument ----
    # Create BRK.A object
    BRK.A <- getSymbols("BRK-A", auto.assign = FALSE)
    
    # Create col_names object with the column names of BRK.A
    col_names <- colnames(BRK.A)
    
    # Set BRK.A column names to syntactically valid names
    colnames(BRK.A) <- colnames(make.names(col_names))
    
    
# 3.7. Create valid names for multiple instruments ----
    # Set name for BRK-A to BRK.A
    setSymbolLookup(BRK.A = list(names = "BRK-A"))
    
    # Set name for T (AT&T) to ATT
    setSymbolLookup(ATT = list(name = "T"))
    
    
    # Load BRK.A and ATT data
    getSymbols(c("BRK.A", "ATT"))
    
    
# 4. Chapter 4: Aligning data with different periodicities   -----
# 4.1. Create an zero-width xts object with a regular index ----
    # Extract the start date of the series
    start_date <- start(irregular_xts)
    
    # Extract the end date of the series
    end_date <- end(irregular_xts)
    
    # Create a regular date-time sequence
    regular_index <- seq(from = start_date,
                         to = end_date,
                         by = "day")
    
    # Create a zero-width xts object
    regular_xts <- xts(NULL, regular_index)
    
    
# 4.2. Merge irregular data with zero-width xts object with regular time index ----
    # Merge irregular_xts and regular_xts
    merged_xts <- merge(irregular_xts, regular_xts)
    
    # Look at the first few rows of merged_xts
    head(merged_xts)
    
    # Use the fill argument to fill NA with their previous value
    merged_filled_xts <- merge(irregular_xts, regular_xts, fill = na.locf)
    
    # Look at the first few rows of merged_filled_xts
    head(merged_filled_xts)
    
    
# 4.3. Aggregate daily series to monthly, convert index to yearmon, merge with monthly series ----
    getSymbols(c("FEDFUNDS", "DFF"), src = "FRED")
    # Aggregate DFF to monthly
    monthly_fedfunds <- apply.monthly(DFF, mean, ra.nr = TRUE)
    
    # Convert index to yearmon
    index(monthly_fedfunds) <- as.yearmon(index(monthly_fedfunds))
    
    # Merge FEDFUNDS with the monthly aggregate
    merged_fedfunds <- merge(FEDFUNDS, monthly_fedfunds)
    
    # Look at the first few rows of the merged object
    head(merged_fedfunds)
    
    
# 4.4. Align aggregated series with first day of month, then last day ----
    getSymbols(c("FEDFUNDS", "DFF"), src = "FRED")
    # Aggregate DFF to monthly
    monthly_fedfunds <- apply.monthly(DFF, mean, ra.nr = TRUE)
    # Merge FEDFUNDS with the monthly aggregate; without converting the monthly_fedfunds index to yearmon first.
    merged_fedfunds <- merge(FEDFUNDS, monthly_fedfunds)
    
    # Look at the first few rows of merged_fedfunds
    head(merged_fedfunds)
    
    # Fill NA forward
    merged_fedfunds_locf <- na.locf(merged_fedfunds)
    
    # Extract index values containing last day of month
    aligned_last_day <- merged_fedfunds_locf[index(monthly_fedfunds)]
    
    # Fill NA backward
    merged_fedfunds_locb <- na.locf(merged_fedfunds, fromLast = TRUE)
    
    # Extract index values containing first day of month
    aligned_first_day <- merged_fedfunds_locb[index(FEDFUNDS)]
    
    
# 4.5. Aggregate to weekly, ending on Wednesdays ----
    # Extract index weekdays
    index_weekdays <- .indexwday(DFF)
    
    # Find locations of Wednesdays
    wednesdays <- which(index_weekdays == 3)
    
    # Create custom end points
    end_points <- c(0, wednesdays, nrow(DFF))
    
    # Calculate weekly mean using custom end points
    weekly_mean <- period.apply(DFF, end_points, mean)
    
    
# 4.6. Combining data that have timezones ----
    datetime <- as.POSIXct("2017-01-18 10:00:00", tz ="UTC")
    london <- xts(1, datetime, tzone ="Europe/London")
    chicago <- xts(1, datetime, tzone ="America/Chicago")

    # Create merged object with a Europe/London timezone
    tz_london <- merge(london, chicago)
    
    # Look at tz_london structure
    str(tz_london)
    
    # Create merged object with a America/Chicago timezone
    tz_chicago <- merge(chicago, london)
    
    # Look at tz_chicago structure
    str(tz_chicago)
    
# 4.7. Making irregular intraday-day data regular ----
    # Create a regular date-time sequence
    regular_index <- seq(as.POSIXct("2010-01-04 09:00"), as.POSIXct("2010-01-08 16:00"), by = "30 min")
    
    # Create a zero-width xts object
    regular_xts <- xts(NULL, regular_index)
    
    # Merge irregular_xts and regular_xts, filling NA with their previous value
    merged_xts <- merge(irregular_xts, regular_index, fill = na.locf)
    
    # Subset to trading day (9AM - 4PM)
    trade_day <- merged_xts["T09:00/T16:00"]
    
    
    
# 4.8. Fill missing values by trading day ----
    # Split trade_day into days
    daily_list <- split(trade_day, f = "days")
    
    # Use lapply to call na.locf for each day in daily_list
    daily_filled <- lapply(daily_list, FUN = na.locf)
    
    # Use do.call to rbind the results
    filled_by_trade_day <- do.call(rbind, daily_filled)
    
    
# 4.9. Aggregate irregular intraday-day data   ----
    # Convert raw prices to 5-second prices
    xts_5sec <- to.period(intraday_xts, period = "seconds", k = 5)
    
    # Convert raw prices to 10-minute prices
    xts_10min <- to.period(intraday_xts, period = "minutes", k = 10)
    
    # Convert raw prices to 1-hour prices
    xts_1hour <- to.period(intraday_xts, period = "hours", k = 1)
    
    
# 5. Chapter 5: Importing text data, and adjusting for corporate actions    -----
# 5.1. Import well-formatted OHLC daily data from text file ----
    # Load AMZN.csv
    getSymbols("AMZN", src = "csv", dir = "")
    
    # Look at AMZN structure
    str(AMZN)
    
# 5.2. Import data from text file ----
    # Import AMZN.csv using read.zoo
    amzn_zoo <- read.zoo("AMZN.csv", sep = ",", header = TRUE)
    
    # Convert to xts
    amzn_xts <- as.xts(amzn_zoo)
    
    # Look at the first few rows of amzn_xts
    head(amzn_xts)
    
    
# 5.3. Handle date and time in separate columns ----
    # The index.column argument is great if your dates and times are in separate columns! 
    # Read data with read.csv
    une_data <- read.csv("UNE.csv", nrows = 5)
    
    # Look at the structure of une_data
    str(une_data)
    
    # Read data with read.zoo, specifying index columns
    une_zoo <- read.zoo("UNE.csv", index.column = c("Date", "Time"), sep = ",", header = TRUE)
    
    # Look at first few rows of data
    head(une_zoo)
    
    
# 5.4. Reading text file that contains multiple instruments ----
    # The two_symbols.csv file in your working directory contains bid/ask data for two instruments, 
    # where each row has one bid or ask observation for one instrument. You will use the split argument to import the data into an object 
    # that has both bid and ask prices for both instruments on one row.
    # Read data with read.csv
    two_symbols_data <- read.csv("two_symbols.csv", nrows = 5)
    
    # Look at the structure of two_symbols_data
    str(two_symbols_data)
    
    # Read data with read.zoo, specifying index columns
    two_symbols_zoo <- read.zoo("two_symbols.csv", split = c("Symbol", "Type"), sep = ",", header = TRUE)
    
    # Look at first few rows of data
    head(two_symbols_zoo)
    
    
# 5.5. Handle missing values ----
    # fill NA using last observation carried forward
    locf <- na.locf(DGS10)
    
    # fill NA using linear interpolation
    approx <- na.approx(DGS10)
    
    # fill NA using spline interpolation
    spline <- na.spline(DGS10)
    
    # merge into one object
    na_filled <- merge(locf, approx, spline)
    
    # plot combined object
    plot(na_filled, col = c("black", "red", "green"))
    
# 5.6. Visualize data ----
    # Download AAPL data from Yahoo Finance
    getSymbols("AAPL", scr = "yahoo")
    
    # Plot close price
    plot(AAPL$AAPL.Close)
    
    # Plot adjusted close price
    plot(AAPL$AAPL.Adjusted)
    
# 5.7. Cross reference sources ----
    aapl_yahoo <- getSymbols("AAPL", scr = "yahoo", auto.assign = FALSE)
    aapl_google <- getSymbols("AAPL", scr = "Google", auto.assign = FALSE)
    
    # Look at first few rows aapl_yahoo
    head(aapl_yahoo)
    
    # Look at first few rows aapl_google
    head(aapl_google)
    
    # Plot difference between Yahoo adjusted close and Google close
    plot(Ad(aapl_yahoo$AAPL.Adjusted) - Cl(aapl_google$AAPL.Close))
    
    # Plot difference between volume from Yahoo and Google
    plot(Vo(aapl_yahoo$AAPL.Volume) - Vo(aapl_google$AAPL.Volume))
    
    
# 5.8. Adjust for stock splits and dividends ----
    # Look at first few rows of AAPL
    head(AAPL)
    
    # Adjust AAPL for splits and dividends
    aapl_adjusted <- adjustOHLC(AAPL)
    
    # Look at first few rows of aapl_adjusted
    head(aapl_adjusted)
    
# 5.9. Download split and dividend data ----
    # Download AAPL split data
    splits <- getSplits("AAPL")
    
    # Print the splits object
    print(splits)
    
    # Download AAPL dividend data
    dividends <- getDividends("AAPL")
    
    # Look at the first few rows of dividends
    head(dividends)
    
    # Download unadjusted AAPL dividend data
    raw_dividends <- getDividends("AAPL", split.adjust = FALSE)
    
    # Look at the first few rows of raw_dividends
    head(raw_dividends)    
    
# 5.10. Adjust univariate data for splits and dividends ----
    # Calculate split and dividend adjustment ratios
    ratios <- adjRatios(splits = splits, dividends = raw_dividends, close = Cl(AAPL))
    
    # Calculate adjusted close for AAPL
    aapl_adjusted <- Cl(AAPL) * ratios[, "Split"] * ratios[, "Div"]
    
    # Look at first few rows of Yahoo adjusted close
    head(Ad(AAPL))
    
    # Look at first few rows of aapl_adjusted
    head(aapl_adjusted)
    