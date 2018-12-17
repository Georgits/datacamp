library(shinydashboard)

sidebar <- dashboardSidebar(
  sidebarMenu(
    # Create two `menuItem()`s, "Dashboard" and "Inputs"
    menuItem("Dashboard", tabName = "dashboard"),
    menuItem("Inputs", tabName = "inputs")
  )
)

body <- dashboardBody(
  # Create a tabBox
  tabItems(
    tabItem(
      tabName = "dashboard",
      tabBox(
        title = "International Space Station Fun Facts",
        tabPanel("Tab1", "Fun Fact 1"),
        tabPanel("Tab2", "Fun Fact 2")
      )
    ),
    tabItem(tabName = "inputs")
  )
  
)


# Use the new sidebar
ui <- dashboardPage(header = dashboardHeader(),
                    sidebar = sidebar,
                    body = body
)

server <- function(input, output) {}

shinyApp(ui, server)