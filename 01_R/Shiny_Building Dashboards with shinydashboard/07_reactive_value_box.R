# Create a reactive value box called click_box that increases in value each time the user clicks the action button.

library(shiny)
library(shinydashboard)

body <- dashboardBody(
  valueBoxOutput("click_box")
)


ui <- dashboardPage(header = dashboardHeader(),
                    sidebar = sidebar,
                    body = body
)


sidebar <- dashboardSidebar(
  actionButton("click", "Update click box")
) 

server <- function(input, output) {
  output$click_box <- renderValueBox({
    valueBox(value = input$click,
             subtitle = "Click Box"
    )
  })
}

shinyApp(ui, server)