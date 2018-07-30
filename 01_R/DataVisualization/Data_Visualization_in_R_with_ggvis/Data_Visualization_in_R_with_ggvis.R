library(ggvis)
library(dplyr)

# Chapter 1: The Grammar of Graphics ----
# Load ggvis and start to explore ----
# Change the code below to plot the disp variable of mtcars on the x axis
mtcars %>% ggvis(~disp, ~mpg) %>% layer_points()


# ggvis and its capabilities ----
# The ggvis packages is loaded into the workspace already

# Change the code below to make a graph with red points
mtcars %>% ggvis(~wt, ~mpg, fill := "red") %>% layer_points()

# Change the code below draw smooths instead of points
mtcars %>% ggvis(~wt, ~mpg) %>% layer_smooths()

# Change the code below to make a graph containing both points and a smoothed summary line
mtcars %>% ggvis(~wt, ~mpg) %>% layer_points() %>% layer_smooths()



# ggvis grammar ~ graphics grammar ----
# Adapt the code: show bars instead of points
pressure %>% ggvis(~temperature, ~pressure) %>% layer_bars()

# Adapt the code: show lines instead of points
pressure %>% ggvis(~temperature, ~pressure) %>% layer_lines()

# Extend the code: map the fill property to the temperature variable
pressure %>% ggvis(~temperature, ~pressure, fill = ~ temperature) %>% layer_points()

# Extend the code: map the size property to the pressure variable
pressure %>% ggvis(~temperature, ~pressure, size = ~ pressure) %>% layer_points()


faithful %>%
  ggvis(~waiting, ~eruptions, fill := "red") %>%
  layer_points() %>%
  add_axis("y", title = "Duration of eruption (m)",
           values = c(2, 3, 4, 5), subdivide = 9) %>%
  add_axis("x", title = "Time since previous eruption (m)")




# Three operators: %>%, = and := ----
# Rewrite the code with the pipe operator
faithful %>% ggvis(~waiting, ~eruptions) %>% layer_points()

# Modify this graph to map the size property to the pressure variable
pressure %>% ggvis(~temperature, ~pressure, size = ~ pressure) %>% layer_points()

# Modify this graph by setting the size property
pressure %>% ggvis(~temperature, ~pressure, size := 100) %>% layer_points()

# Fix this code to set the fill property to red
pressure %>% ggvis(~temperature, ~pressure, fill := "red") %>% layer_points()



# Referring to different objects ----
# Which of the commands below will create a graph that has green points? 
red <- "green"
pressure$red <- pressure$temperature

# GRAPH A
pressure %>%
  ggvis(~temperature, ~pressure,
        fill = ~red) %>%
  layer_points()

# GRAPH B
pressure %>%
  ggvis(~temperature, ~pressure,
        fill = "red") %>%
  layer_points()

# GRAPH C
pressure %>%
  ggvis(~temperature, ~pressure,
        fill := red) %>%
  layer_points()



# Referring to different objects (2) ----
# Which of the commands below will create a graph that uses color to reveal 
# the values of the temperature variable in the pressure data set?
red <- "green"
pressure$red <- pressure$temperature

# GRAPH A
pressure %>%
  ggvis(~temperature, ~pressure,
        fill = ~red) %>%
  layer_points()

# GRAPH B
pressure %>%
  ggvis(~temperature, ~pressure,
        fill = "red") %>%
  layer_points()

# GRAPH C
pressure %>%
  ggvis(~temperature, ~pressure,
        fill := red) %>%
  layer_points()



# Properties for points ----
# You can manipulate many different properties when using 
# layer_points(), including x, y, fill, fillOpacity, opacity, shape, size, stroke, strokeOpacity, andstrokeWidth.

# The shape property recognizes several different values: 
# circle (default), square, cross, diamond, triangle-up, and triangle-down.

# Add code
faithful %>% ggvis(~waiting, ~eruptions, 
                   size = ~eruptions, 
                   opacity := 0.5, 
                   fill := "blue", 
                   stroke := "black") %>% 
  layer_points()

# Add code
faithful %>% ggvis(~waiting, ~eruptions, 
                   fillOpacity = ~ eruptions, 
                   size := 100, 
                   fill := "red", 
                   stroke := "red", 
                   shape := "cross") %>% 
  layer_points()




#Properties for lines ----
# Update the code
pressure %>% ggvis(~temperature, ~pressure, stroke := "red", strokeWidth := 2, strokeDash := 6) %>% layer_lines()



# Path marks and polygons ----
# Update the plot
texas %>% ggvis(~long, ~lat, fill := "darkorange") %>% layer_paths()



# Display model fits ----
# Compute the x and y coordinates for a loess smooth line that predicts mpg with the wt
mtcars %>% compute_smooth(mpg ~ wt)




# compute_smooth() to simplify model fits -----
# Extend with ggvis() and layer_lines()
mtcars %>% compute_smooth(mpg ~ wt) %>% ggvis(~pred_, ~resp_) %>% layer_lines()

# Extend with layer_points() and layer_smooths()
mtcars %>% ggvis(~wt, ~mpg) %>% layer_points() %>% layer_smooths()
# Because first calling compute_smooth() and then layer_lines() can be a bit of a hassle, 
# ggvis features the layer_smooths() function: this layer automatically calls compute_smooth() in the background and plots 
# the results as a smoothed line.




# Chapter 3: Transformations -----
# Histograms (1) ----
# Build a histogram with a binwidth of 5 units
faithful %>% ggvis(~waiting) %>% layer_histograms(width = 5)


# Histograms (2) ----
# Finish the command
faithful %>%
  compute_bin(~waiting, width = 5) %>%
  ggvis(x = ~xmin_, x2 = ~xmax_, y = 0, y2 = ~count_) %>%
  layer_rects()
# Well done! This is the same plot that you've coded up in the previous exercise.
# Remember that there is no need to explicitly code the combination of compute_bin() and layer_rects(). 
# Unless you want to do special things, use layer_histograms().



# Density plots ----
# Build the density plot
faithful %>% ggvis(~waiting, fill := "green") %>% layer_densities()



# !!!! Shortcuts -----
# Instead of compute_smooth() and layer_lines(), you can use layer_smooths()
# Instead of compute_bin() and layer_rects(), you can use layer_histograms()
# Instead of compute_density() and layer_lines(), you can use layer_densities()


# Simplify the code
mtcars %>%
  compute_count(~factor(cyl)) %>%
  ggvis(x = ~x_, y = 0, y2 = ~count_, width = band()) %>%
  layer_rects()

# Simplified code
mtcars %>%
  ggvis(~factor(cyl)) %>%
  layer_bars()




# ggvis and group_by ----
# Instruction 1
mtcars %>% group_by(cyl) %>% ggvis(~mpg, ~wt, stroke = ~factor(cyl)) %>% layer_smooths()

# Instruction 2
mtcars %>% group_by(cyl) %>% ggvis(~mpg, fill = ~factor(cyl)) %>% layer_densities()




# group_by() versus interaction() ----
# Alter the graph
mtcars %>% group_by(cyl, am) %>% ggvis(~mpg, fill = ~interaction(cyl, am)) %>% layer_densities()
# Make sure you do not mix up group_by() and interaction(). The former is used to group observations, 
# while the latter allows you to specify properties.



# Chaining is a virtue ----
mtcars %>%
  group_by(am) %>%
  ggvis(~mpg, ~hp) %>%
  layer_smooths(stroke = ~factor(am)) %>%
  layer_points(fill = ~factor(am))


mtcars %>%
  group_by(am) %>%
  ggvis(~mpg, ~hp) %>%
  layer_points(fill = ~factor(am)) %>% 
  layer_smooths(stroke = ~factor(am))




# Interactivity and Layers ----
# Adapt the code: set fill with a select box
faithful %>%
  ggvis(~waiting, ~eruptions, fillOpacity := 0.5,
        shape := input_select(label = "Choose shape:",
                              choices = c("circle", "square", "cross",
                                          "diamond", "triangle-up", "triangle-down")),
        fill := input_select(label = "Choose color:",
                             choices = c("black", "red", "blue", "green"))) %>%
  layer_points()

# Add radio buttons to control the fill of the plot
mtcars %>%
  ggvis(~mpg, ~wt,
        fill := input_radiobuttons(label = "Choose color:",
                                   choices = c("black", "red", "blue", "green"))) %>%
  layer_points()



# Input widgets in more detail ----
# Change the radiobuttons widget to a text widget
mtcars %>%
  ggvis(~mpg, ~wt,
        fill := input_text(label = "Choose color:",
                           value = "black")) %>%
  layer_points()



# Input widgets in more detail (2) ----
# Map the fill property to a select box that returns variable names
mtcars %>%
  ggvis(~mpg, ~wt,
        fill = input_select(label = "Choose fill variable:",
                            choices = names(mtcars),
                            map = as.name)) %>%
  layer_points()



# Control parameters and values ----
# Map the bindwidth to a numeric field ("Choose a binwidth:")
mtcars %>%
  ggvis(~mpg) %>%
  layer_histograms(
    width = input_numeric(label = "Choose a binwidth:",
                          value = 1))


# Control parameters and values ----
# Map the binwidth to a slider bar ("Choose a binwidth:") with the correct specifications
mtcars %>%
  ggvis(~mpg) %>%
  layer_histograms(width = input_slider(label = "Choose a binwidth:",
                                        min = 1, max = 20,
                                        value = 11))




# Multi-layered plots and their properties ----
# Add a layer of points to the graph below.
pressure %>%
  ggvis(~temperature, ~pressure, stroke := "skyblue") %>%
  layer_lines() %>%
  layer_points()

# Copy and adapt the solution to the first instruction below so that only the lines layer uses a skyblue stroke.
pressure %>%
  ggvis(~temperature, ~pressure) %>%
  layer_lines(stroke := "skyblue") %>%
  layer_points()



# Multi-layered plots and their properties (2) ---
# Rewrite the code below so that only the points layer uses the shape property.
pressure %>%
  ggvis(~temperature, ~pressure) %>%
  layer_lines(stroke := "skyblue") %>%
  layer_points(shape := "triangle-up")

# Refactor the code for the graph below to make it as concise as possible
pressure %>%
  ggvis(~temperature, ~pressure, stroke := "skyblue",
        strokeOpacity := 0.5, strokeWidth := 5) %>%
  layer_lines() %>%
  layer_points(fill = ~temperature, size := 300,
               shape := "triangle-up")




# There is no limit on the number of layers! ----
# Add more layers to the line plot
# Add more layers to the line plot
pressure %>%
  ggvis(~temperature, ~pressure) %>%
  layer_lines(opacity := 0.5) %>%
  layer_points() %>%
  layer_model_predictions(model = "lm", stroke := "navy") %>%
  layer_smooths(stroke := "skyblue")



# Taking local and global to the next level ----
pressure %>%
  ggvis(~temperature, ~pressure, stroke := "darkred") %>%
  layer_lines(stroke := "orange", strokeDash := 5, strokeWidth := 5) %>%
  layer_points(size := 100, fill := "lightgreen", shape := "circle") %>%
  layer_smooths()




# Chapter 5: Customizing Axes, Legends, and Scales ----
# Axes ----
# Change the axes of the plot as instructed
faithful %>% 
  ggvis(~waiting, ~eruptions) %>% 
  layer_points() %>%
  add_axis("x", 
           title = "Time since previous eruption (m)", 
           values = c(50, 60, 70, 80, 90), 
           subdivide = 9,
           orient = "top") %>%
  add_axis("y", 
           title = "Duration of eruption (m)", 
           values = c(2, 3, 4, 5), 
           subdivide = 9,
           orient = "right")



# Legends ----
# Add a legend
faithful %>% 
  ggvis(~waiting, ~eruptions, opacity := 0.6, 
        fill = ~factor(round(eruptions))) %>% 
  layer_points() %>%
  add_legend("fill", title = "~ duration (m)", orient = "left")


# Legends (2) ----
# Fix the legend
faithful %>% 
  ggvis(~waiting, ~eruptions, opacity := 0.6, 
        fill = ~factor(round(eruptions)), shape = ~factor(round(eruptions)), 
        size = ~round(eruptions))  %>%
  layer_points() %>%
  add_legend(c("fill", "shape", "size"), title = "~ duration (m)")




# Scale types ----
# Add a scale_numeric()
mtcars %>% 
  ggvis(~wt, ~mpg, fill = ~disp, stroke = ~disp, strokeWidth := 2) %>%
  layer_points() %>%
  scale_numeric("fill", range = c("red", "yellow")) %>%
  scale_numeric("stroke", range = c("darkred", "orange"))

# Add a scale_numeric()
mtcars %>% ggvis(~wt, ~mpg, fill = ~hp) %>%
  layer_points() %>%
  scale_numeric("fill", range = c("green", "beige"))

# Add a scale_nominal()
mtcars %>% ggvis(~wt, ~mpg, fill = ~factor(cyl)) %>%
  layer_points() %>%
  scale_nominal("fill", range = c("purple", "blue", "green"))




# Adjust any visual property -----
# Add a scale to limit the range of opacity 
mtcars %>% ggvis(x = ~wt, y = ~mpg, fill = ~factor(cyl), opacity = ~hp) %>%
  layer_points() %>%
  scale_numeric("opacity", range = c(0.2, 1))



# Adjust any visual property (2) ----
# Add a second scale to set domain for x
mtcars %>% ggvis(~wt, ~mpg, fill = ~disp) %>%
  layer_points() %>%
  scale_numeric("y", domain = c(0, NA)) %>%
  scale_numeric("x", domain = c(0,6))





# "=" versus ":=" ----
# We prepared a new version of mtcars that contains a color column with valid color names
# mtcars contains column color
mtcars %>% 
  ggvis(x = ~wt, y = ~mpg, fill = ~color) %>% 
  layer_points()

# Set the fill instead of mapping it
mtcars %>% 
  ggvis(x = ~wt, y = ~mpg, fill := ~color) %>% 
  layer_points()
