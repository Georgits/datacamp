# testing out having a shared script that defines useful setup functions

library(ggplot2)

read_dataset <- function(file) {
  url <- paste0("http://s3.amazonaws.com/assets.datacamp.com/production/course_2351/datasets/",
                file, ".rds")
  download.file(url, "temp.rds")
  readRDS("temp.rds")
}

compare_histograms <- function(variable1, variable2) {
  x <- data.frame(value = variable1, variable = "Variable 1")
  y <- data.frame(value = variable2, variable = "Variable 2")
  ggplot(rbind(x, y), aes(value)) +
    geom_histogram() +
    facet_wrap(~ variable, nrow = 2)
}
