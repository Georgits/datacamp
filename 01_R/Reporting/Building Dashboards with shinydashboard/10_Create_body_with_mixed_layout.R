library(shiny)
library(shinydashboard)

body <- dashboardBody(
  # Update the CSS
  tags$head(tags$style(HTML('h3 {font-weight: bold;}'))),
  # tags$style(HTML('div {font-weight: bold;}')),
  fluidRow(
    # Row 1
    box(
      width = 12,
      title = "Regular Box, Row 1",
      "Star Wars, nothing but Star Wars"
      )
    ),
  fluidRow(
    # Row 2, Column 1
    column(width = 6,
           infoBox(
             width = NULL,
             title = "Regular Box, Row 2, Column 1",
             subtitle = "Gimme those Star Wars"
             )
           ),
    # Row 2, Column 2
    column(width = 6,
           infoBox(
             width = NULL,
             title = "Regular Box, Row 2, Column 2",
             subtitle = "Don't let them end"
             )
           )
    )
  )

ui <- dashboardPage(
  skin = "purple",
  header = dashboardHeader(),
  sidebar = dashboardSidebar(),
  body = body
)

server <- function(input, output) {}

shinyApp(ui, server)