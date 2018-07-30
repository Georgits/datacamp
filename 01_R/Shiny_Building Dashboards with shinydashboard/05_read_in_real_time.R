library(shiny)
library(shinydashboard)
starwars_url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_6225/datasets/starwars.csv"


body <- dashboardBody(
  tableOutput("table")
)

ui <- dashboardPage(header = dashboardHeader(),
                    sidebar = dashboardSidebar(),
                    body = body
)


server <- function(input, output, session) {
  reactive_starwars_data <- reactiveFileReader(
    intervalMillis = 1000,
    session = session,
    filePath = starwars_url,
    readFunc = function(filePath) { 
      read.csv(url(filePath))
    }
  )
  
  output$table <- renderTable({
    reactive_starwars_data()
  })
}



shinyApp(ui, server)