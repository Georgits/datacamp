library(birdnik)
library(httr)
library(jsonlite)
library(rlist)
library(dplyr)
library(xml2)
library(rvest)


# Chapter 1:  Downloading Files and Using API Clients -----

# Downloading files and reading them into R
  # Here are the URLs! As you can see they're just normal strings
  csv_url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_1561/datasets/chickwts.csv"
  tsv_url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_3026/datasets/tsv_data.tsv"
  
  # Read a file in from the CSV URL and assign it to csv_data
  csv_data <- read.csv(csv_url)
  
  # Read a file in from the TSV URL and assign it to tsv_data
  tsv_data <- read.delim(tsv_url)
  
  # Examine the objects with head()
  head(csv_data)
  head(tsv_data)

  
  
  
# Saving raw files to disk
  # Download the file with download.file()
  download.file(url = csv_url, destfile = "feed_data.csv")
  
  # Read it in with read.csv()
  csv_data <- read.csv("feed_data.csv")



# Saving formatted files to disk  
  # Add a new column: square_weight
  csv_data$square_weight <- csv_data$weight^2
  
  # Save it to disk with saveRDS()
  saveRDS(csv_data, "modified_feed_data.RDS")
  
  # Read it back in with readRDS()
  modified_feed_data <- readRDS("modified_feed_data.RDS")
  
  # Examine modified_feed_data
  str(modified_feed_data)
  
  
  
# Using API clients
  # Load pageviews
  library(pageviews)
  
  # Get the pageviews for "Hadley Wickham"
  hadley_pageviews <- article_pageviews(project = "en.wikipedia", "Hadley Wickham")
  
  # Examine the resulting object
  str(hadley_pageviews)
  
  
# Using access tokens
  # Get the word frequency for "vector", using api_key to access it
  api_key = "d8ed66f01da01b0c6a0070d7c1503801993a39c126fbc3382"
  vector_frequency <- word_frequency(api_key, "vector")
  
  
  
  
  
  
  
  
  
  
  

# Chapter 2: Using httr to interact with APIs directly -----

# GET requests in practice
  # Make a GET request to http://httpbin.org/get
  get_result <- GET('http://httpbin.org/get')
  
  # Print it to inspect it
  get_result
  
  
# POST requests in practice  
  # Load the httr package
  library(httr)
  
  # Make a POST request to http://httpbin.org/post with the body "this is a test"
  post_result <- POST(url = 'http://httpbin.org/post', body = "this is a test")
  
  # Print it to inspect it
  post_result
  
  
# Extracting the response
  url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/Hadley_Wickham/daily/20170101/20170102"
  # Make a GET request to url and save the results
  pageview_response <- GET(url = url)
  
  # Call content() to retrieve the data the server sent back
  pageview_data <- content(pageview_response)
  
  # Examine the results with str()
  str(pageview_data)
  
  
  
  
  
# Handling http failures  
  fake_url <- "http://google.com/fakepagethatdoesnotexist"
  
  # Make the GET request
  request_result <- GET(url = fake_url)
  
  # Check request_result
  if (http_error(request_result)) {
    warning("The request failed")
  } else {
    content(request_result)
  } 
  
  
  
# Constructing queries (Part I)
  # Construct a directory-based API URL to `http://swapi.co/api`,
  # looking for person `1` in `people`
  directory_url <- paste("http://swapi.co/api", "people", "1", sep = "/")
  
  # Make a GET call with it
  result <- GET(directory_url)
  
  
  
# Constructing queries (Part II)
  # Create list with nationality and country elements
  query_params <- list(nationality = "americans", 
                       country = "antigua")
  
  # Make parameter-based call to httpbin, with query_params
  parameter_response <- GET("https://httpbin.org/get", query = query_params)
  
  # Print parameter_response
  print(parameter_response)
  
  
  
#  Using user agents
  # Do not change the url
  url <- "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Aaron_Halfaker/daily/2015100100/2015103100"
  
  # Add the email address and the test sentence inside user_agent()
  server_response <- GET(url, user_agent("my@email.addres this is a test"))
  
  
  
# Rate-limiting
  # Construct a vector of 2 URLs
  urls <- c("http://fakeurl.com/api/1.0/", "http://fakeurl.com/api/2.0/")
  
  for (url in urls) {
    # Send a GET request to url
    result <- GET(url)
    # Delay for 5 seconds between requests
    Sys.sleep(5)
  }
  
  
#  Tying it all together
  
  get_pageviews <- function(article_title){
    
    url <- paste0("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents", 
                  article_title, 
                  "daily/2015100100/2015103100", sep = "/") 
    
    response <- GET(url, user_agent("my@email.com this is a test")) 
    
    if (http_error(response)){ 
      
      stop("the request failed") 
      
    } else { 
      
      result <- content(response) 
      
      return(result) 
      
    }
  }
  
  
  
  
# Chapter 3: Handling JSON and XML ----
  # helper-function
  rev_history <- function(title, format = "json"){
    if (title != "Hadley Wickham") {
      stop('rev_history() only works for `title = "Hadley Wickham"`')
    }
    if (format == "json") {
        resp <- readRDS("had_rev_json.rds")
    } else if (format == "xml") {
      resp <- readRDS("had_rev_xml.rds")
    } else {
      stop('Invalid format supplied, try "json" or "xml"')
    }
    resp  
  }  
  
  # Parsing JSON
  # Get revision history for "Hadley Wickham"
  resp_json <- rev_history("Hadley Wickham")
  
  # Check http_type() of resp_json
  http_type(resp_json)
  
  # Examine returned text with content()
  content(resp_json, as = "text")
  
  # Parse response with content()
  content(resp_json, as = "parsed")
  
  # Parse returned text with fromJSON()
  fromJSON(content(resp_json, as = "text"))
  
  
  
  
  # Manipulating parsed JSON

  resp_json <- GET("https://en.wikipedia.org/w/api.php?action=query&titles=Hadley%20Wickham&prop=revisions&rvprop=timestamp%7Cuser%7Ccomment%7Ccontent&rvlimit=5&format=json&rvdir=newer&rvstart=2015-01-14T17%3A12%3A45Z&rvsection=0")
  # Examine output of this code
  str(content(resp_json), max.level = 4)
  
  # Store revision list
  revs <- content(resp_json)$query$pages$`41916270`$revisions
  
  # Extract the user element
  user_time <- list.select(revs, user, timestamp)
  
  # Print user_time
  print(user_time)
  
  # Stack to turn into a data frame
  list.stack(user_time)
  
  
  
  # Reformatting JSON

  # Pull out revision list
  revs <- content(resp_json)$query$pages$`41916270`$revisions
  
  # Extract user and timestamp
  revs %>%
    bind_rows() %>%           
    select(user, timestamp)
  
  
  # Examining XML documents
  # helper-function
  rev_history <- function(title, format = "json"){
    if (title != "Hadley Wickham") {
      stop('rev_history() only works for `title = "Hadley Wickham"`')
    }
    
    if (format == "json") {
      resp <- readRDS("had_rev_json.rds")
    } else if (format == "xml") {
      resp <- readRDS("had_rev_xml.rds")
    } else {
      stop('Invalid format supplied, try "json" or "xml"')
    }
    resp  
  }
  # Get XML revision history
  resp_xml <- rev_history("Hadley Wickham", format = "xml")

  # Check response is XML 
  http_type(resp_xml)
  
  # Examine returned text with content()
  rev_text <- content(resp_xml, as = "text")
  rev_text
  
  # Turn rev_text into an XML document
  rev_xml <- read_xml(rev_text)
  
  # Examine the structure of rev_xml
  xml_structure(rev_xml)
  
  
  
  # Extracting XML data
  # Find all nodes using XPATH "/api/query/pages/page/revisions/rev"
  xml_find_all(rev_xml, "/api/query/pages/page/revisions/rev")
  
  # Find all rev nodes anywhere in document
  rev_nodes <- xml_find_all(rev_xml, "//rev")
  
  # Use xml_text() to get text from rev_nodes
  xml_text(rev_nodes)
  
  
  
  # Extracting XML attributes
  
  # Be careful with the difference between the singular (i.e. without the s, xml_attr()) and plural (xml_attrs()) functions. 
  # If you are extracting a specific named attribute you'll always use the singular xml_attr() and 
  # need to supply two arguments: the XML nodeset and the name of the attribute.
  
  # All rev nodes
  rev_nodes <- xml_find_all(rev_xml, "//rev")
  
  # The first rev node
  first_rev_node <- xml_find_first(rev_xml, "//rev")
  
  # Find all attributes with xml_attrs()
  xml_attrs(first_rev_node)
  
  # Find user attribute with xml_attr()
  xml_attr(first_rev_node, "user")
  
  # Find user attribute for all rev nodes
  xml_attr(rev_nodes, "user")
  
  # Find anon attribute for all rev nodes
  xml_attr(rev_nodes, "anon")
  
  
  
  # Wrapup: returning nice API output
  get_revision_history <- function(article_title){
    # Get raw revision response
    rev_resp <- rev_history(article_title, format = "xml")
    
    # Turn the content() of rev_resp into XML
    rev_xml <- read_xml(content(rev_resp, "text"))
    
    # Find revision nodes
    rev_nodes <- xml_find_all(rev_xml, "//rev")
    
    # Parse out usernames
    user <- xml_attr(rev_nodes, "user")
    
    # Parse out timestamps
    timestamp <- readr::parse_datetime(xml_attr(rev_nodes, "timestamp"))
    
    # Parse out content
    content <- xml_text(rev_nodes)
    
    # Return data frame 
    data.frame(user = user,
               timestamp = timestamp,
               content = substr(content, 1, 40))
  }
  
  # Call function for "Hadley Wickham"
  get_revision_history("Hadley Wickham")
  
  
  
  
# Chapter 4: Web scraping with XPATHs ----
  # Reading HTML
  # Hadley Wickham's Wikipedia page
  test_url <- "https://en.wikipedia.org/wiki/Hadley_Wickham"
  
  # Read the URL stored as "test_url" with read_html()
  test_xml <- read_html(test_url)
  
  # Print test_xml
  test_xml
  
  
  
  # Extracting nodes by XPATH
  test_node_xpath <- "//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"vcard\", \" \" ))]"
  
  # Use html_node() to grab the node with the XPATH stored as `test_node_xpath`
  node <- html_node(x = test_xml, xpath = test_node_xpath)
  
  # Print the first element of the result
  node[1]
  
  
  
  # Extracting names
  # Extract the name of table_element
  element_name <- html_name(node)
  
  # Print the name
  element_name
  
  
  
  # Extracting values
  second_xpath_val <- "//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"fn\", \" \" ))]"
  
  # Extract the element of table_element referred to by second_xpath_val and store it as page_name
  page_name <- html_node(x = node, xpath = second_xpath_val)
  
  # Extract the text from page_name
  page_title <- html_text(page_name)
  
  # Print page_title
  page_title
  
  
  
  
  # Extracting tables
  # Turn table_element into a data frame and assign it to wiki_table
  wiki_table <- html_table(node)
  
  # Print wiki_table
  wiki_table
  
  
  
  # Cleaning a data frame
  # Rename the columns of wiki_table
  colnames(wiki_table) <- c("key", "value")
  
  # Remove the empty row from wiki_table
  cleaned_table <- subset(wiki_table, (!key == "" | !value == ""))
  
  # Print cleaned_table
  cleaned_table
  
  
  
  
# Chapter 5: CSS web scraping in theory ----
  # Using CSS to scrape nodes
  test_url <- "https://en.wikipedia.org/wiki/Hadley_Wickham"
  test_xml <- read_html(test_url)
  
  # Select the table elements
  html_nodes(test_xml, css = "table")
  
  # Select elements with class = "infobox"
  html_nodes(test_xml, css = ".infobox")
  
  # Select elements with id = "firstHeading"
  html_nodes(test_xml, css = "#firstHeading")
  
  
  
  # Scraping names
  # Extract element with class infobox
  infobox_element <- html_nodes(test_xml, css = ".infobox")
  
  # Get tag name of infobox_element
  element_name <- html_name(infobox_element)
  
  # Print element_name
  element_name
  
  
  
  # Scraping text
  # Extract element with class fn
  page_name <- html_node(x = infobox_element, css = ".fn")
  
  # Get contents of page_name
  page_title <- html_text(page_name)
  
  # Print page_title
  page_title
  
  
  
  
  # API calls
  # Load httr
  library(httr)
  
  # The API url
  base_url <- "https://en.wikipedia.org/w/api.php"
  
  # Set query parameters
  query_params <- list(action = "parse", 
                       page = "Hadley Wickham", 
                       format = "xml")
  
  # Get data from API
  resp <- GET(url = base_url, query = query_params)
  
  # Parse response
  resp_xml <- content(resp)
  
  
  
  # Extracting information
  # Load rvest
  library(rvest)
  
  # Read page contents as HTML
  page_html <- read_html(xml_text(resp_xml))
  
  # Extract infobox element
  infobox_element <- html_node(page_html, css = ".infobox")
  
  # Extract page name element from infobox
  page_name <- html_node(infobox_element, css = ".fn")
  
  # Extract page name as text
  page_title <- html_text(page_name)
  
  
  
  # Normalising information
  # Your code from earlier exercises
  wiki_table <- html_table(infobox_element)
  colnames(wiki_table) <- c("key", "value")
  cleaned_table <- subset(wiki_table, !key == "")
  
  # Create a dataframe for full name
  name_df <- data.frame(key = "Full name", value = page_title)
  
  # Combine name_df with cleaned_table
  wiki_table2 <- rbind(name_df, cleaned_table)
  
  # Print wiki_table
  wiki_table2
  
  
  
  # Reproducibility
  library(httr)
  library(rvest)
  library(xml2)
  
  get_infobox <- function(title){
    base_url <- "https://en.wikipedia.org/w/api.php"
    
    # Change "Hadley Wickham" to title
    query_params <- list(action = "parse", 
                         page = title, 
                         format = "xml")
    
    resp <- GET(url = base_url, query = query_params)
    resp_xml <- content(resp)
    
    page_html <- read_html(xml_text(resp_xml))
    infobox_element <- html_node(x = page_html, css =".infobox")
    page_name <- html_node(x = infobox_element, css = ".fn")
    page_title <- html_text(page_name)
    
    wiki_table <- html_table(infobox_element)
    colnames(wiki_table) <- c("key", "value")
    cleaned_table <- subset(wiki_table, !wiki_table$key == "")
    name_df <- data.frame(key = "Full name", value = page_title)
    wiki_table <- rbind(name_df, cleaned_table)
    
    wiki_table
  }
  
  # Test get_infobox with "Hadley Wickham"
  get_infobox(title = "Hadley Wickham")
  
  # Try get_infobox with "Ross Ihaka"
  get_infobox(title = "Ross Ihaka")
  
  # Try get_infobox with "Grace Hopper"
    get_infobox(title = "Grace Hopper")