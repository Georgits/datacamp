# Winston Chang and Joe Cheng Building Dashboards with Shiny Tutorial
# https://www.rstudio.com/resources/videos/building-dashboards-with-shiny-tutorial/

# Winston Chang Dynamic Dashboards with Shiny 
# https://www.rstudio.com/resources/webinars/dynamic-dashboards-with-shiny/

# https://cneos.jpl.nasa.gov/fireballs/

library(readr)
library(dplyr)
library(tidyverse)
library(shiny)
library(shinydashboard)
library(leaflet)
load("C:/Users/D91067/Desktop/R/Building Dashboards with shinydashboard/nasa_fireball.rda")

# nasa_fireball <- read_csv("cneos_fireball_data.csv")
# nasa_fireball <- nasa_fireball %>%
#   drop_na(`Longitude (deg.)`, `Latitude (deg.)`) %>% 
#   rename(lon = `Longitude (deg.)`, lat = `Latitude (deg.)`) 
# 
# nasa_fireball$lat_dir <- substr(nasa_fireball$lat, nchar(nasa_fireball$lat)-1, nchar(nasa_fireball$lat))
# nasa_fireball$lon_dir <- substr(nasa_fireball$lon, nchar(nasa_fireball$lon)-1, nchar(nasa_fireball$lon))
# 
# nasa_fireball$lat <- substr(nasa_fireball$lat, 1, nchar(nasa_fireball$lat)-1)
# nasa_fireball$lon <- substr(nasa_fireball$lon, 1, nchar(nasa_fireball$lon)-1)
# 
# 
# nasa_fireball$lat <- as.numeric(nasa_fireball$lat)
# nasa_fireball$lon <- as.numeric(nasa_fireball$lon)


# 1. Examine Dataset ----
# Print the nasa_fireball data frame
print(nasa_fireball)

# Examine the types of variables present
sapply(nasa_fireball, class)

# Observe the number of observations in this data frame
nrow(nasa_fireball)

# Check for missing data
sapply(nasa_fireball, anyNA)


max_vel <- max(nasa_fireball$vel, na.rm = TRUE)
max_impact_e <- max(nasa_fireball$impact_e, na.rm = TRUE)
max_energy <- max(nasa_fireball$energy, na.rm = TRUE)
n_us <- sum(
  ifelse(
    nasa_fireball$lat < 64.9 & nasa_fireball$lat > 19.5
    & nasa_fireball$lon < -68.0 & nasa_fireball$lon > -161.8,
    1, 0),
  na.rm = TRUE)



# 2. App ----

sidebar <- dashboardSidebar(
  sliderInput(
    inputId = "threshold",
    label = "Color Threshold",
    min = 0, max = 100, value = 10)
)


body <- dashboardBody(
  fluidRow(
    # Add a value box for maximum energy
    valueBox(
      value = max_energy, 
      subtitle = "Maximum total radiated energy (Joules)",
      icon = icon("lightbulb-o")
    ),
    # Add a value box for maximum impact
    valueBox(
      value = max_impact_e,
      subtitle = "Maximum impact energy (kilotons of TNT)", 
      icon = icon("star")
    ),
    # Add a value box for maximum velocity
    valueBox(
      value = max_vel,
      subtitle = "Maximum pre-impact velocity",
      icon = icon("fire")
    ),
    fluidRow(
      valueBoxOutput("us_box")
    ),
    fluidRow(
      leafletOutput("plot")
    )
  )
)

ui <- dashboardPage(header = dashboardHeader(),
                    sidebar = sidebar,
                    body = body
)


server <- function(input, output) {
  output$us_box <- renderValueBox({
    valueBox(
      value = n_us,
      subtitle = "Number of Fireballs in the US",
      icon = icon("globe"),
      color = if (n_us < input$threshold) {
        "blue"
      } else {
        "fuchsia"
      }
    )
  })
  
  output$plot <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%  
      addCircleMarkers(
        lng = nasa_fireball$lon, 
        lat = nasa_fireball$lat, 
        radius = log(nasa_fireball$impact_e), 
        label = nasa_fireball$date, 
        weight = 2)
  })
  
}



shinyApp(ui, server)