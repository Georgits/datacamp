library(shiny)
library(shinydashboard)

header <- dashboardHeader(
  dropdownMenu(
    type = "notifications",
    notificationItem(
      text = "The International Space Station is overhead!",
      icon = icon("rocket")
    )
  )
)

body <- dashboardBody(
  tags$head(
    tags$style(HTML('h3 {font-weight: bold;}'))),
  fluidRow(
    box(
      width = 12,
      title = "Regular Box, Row 1",
      "Star Wars, nothing but Star Wars",
      # Make the box red
      status = "danger"
    )
  ),
  fluidRow(
    column(width = 6,
           infoBox(
             width = NULL,
             title = "Regular Box, Row 2, Column 1",
             subtitle = "Gimme those Star Wars",
             # Add a star icon
             icon = icon("star")
           )
    ),
    column(width = 6,
           infoBox(
             width = NULL,
             title = "Regular Box, Row 2, Column 2",
             subtitle = "Don't let them end",
             # Make the box yellow
             color = "yellow"
           )
    )
  )
  )

ui <- dashboardPage(
  skin = "purple",
  header = header,
  sidebar = dashboardSidebar(),
  body = body)
shinyApp(ui, server)