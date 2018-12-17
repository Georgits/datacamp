ui <- fluidPage(
  numericInput("num1", "Number 1", 5),
  numericInput("num2", "Number 2", 10),
  textOutput("result")
)

server <- function(input, output) {
  # Calculate the sum of the inputs
  sum <- reactive({
    input$num1 + input$num2
  })
  
  # Calculate the average of the inputs
  average <- reactive({
    sum() / 2
  })
  
  output$result <- renderText({
    paste(
      # Print the calculated sum
      "The sum is", sum(),
      # Print the calculated average
      "and the average is", average()
    )
  })
}

shinyApp(ui, server)