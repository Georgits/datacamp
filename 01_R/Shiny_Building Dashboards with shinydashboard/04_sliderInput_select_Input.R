library(shinydashboard)
library(dplyr)
data("starwars")

header <- dashboardHeader()
body <- dashboardBody()


sidebar <- dashboardSidebar(
  # Add a slider
  sliderInput(
    inputId = "height",
    label = "Height",
    min = 66, max = 264,
    value = 264),
  
  # Create a select list
  selectInput(
    inputId = "name",
    label = "Name",
    choices = starwars$name
  )
  
)

body <- dashboardBody(
  textOutput("name1")
)


ui <- dashboardPage(header = dashboardHeader(),
                    sidebar = sidebar,
                    body = body
)

server <- function(input, output) {
  output$name1 <- renderText({
    input$name
  })
}

shinyApp(ui, server) 
