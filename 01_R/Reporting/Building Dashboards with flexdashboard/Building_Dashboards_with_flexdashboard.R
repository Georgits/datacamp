library(shinydashboard)

header <- dashboardHeader()
sidebar <- dashboardSidebar()
body <- dashboardBody()

# Create the UI using the header, sidebar, and body
ui <- dashboardPage(header, sidebar, body)

server <- function(input, output) {}

shinyApp(ui, server)



# Chaoter 1:  Dashboard Layouts
