library(MASS)


# 4. Chapter 4: ---- 
# Constructing and displaying layout matrices
  # Define row1, row2, row3 for plots 1, 2, and 3
  row1 <- c(0, 1)
  row2 <- c(2, 0)
  row3 <- c(0, 3)
  
  # Use the matrix function to combine these rows into a matrix
  layoutMatrix <- matrix(c(row1, row2, row3), 
                         byrow = TRUE, nrow = 3)
  
  # Call the layout() function to set up the plot array
  layout(layoutMatrix)
  
  # Show where the three plots will go 
  layout.show(3)


# Constructing and displaying layout matrices
  # Create row1, row2, and layoutVector
  row1 <- c(1, 0, 0)
  row2 <- c(0, 2, 2)
  layoutVector <- c(row1,rep(row2, 2))
  
  # Convert layoutVector into layoutMatrix
  layoutMatrix <- matrix(layoutVector, byrow = TRUE, nrow = 3)
  
  # Set up the plot array
  layout(layoutMatrix )
  
  # Plot scatterplot
  plot(Boston$rad, Boston$zn)
  
  # Plot sunflower plot
  sunflowerplot(Boston$rad, Boston$zn)
  

# Creating arrays with different sized plots
  # Set up the plot array
  layout(layoutMatrix)
  
  # Construct vectors indexB and indexA
  indexB <- which(whiteside$Insul == "Before")
  indexA <- which(whiteside$Insul == "After")
  
  # Create plot 1 and add title
  plot(whiteside$Temp[indexB], whiteside$Gas[indexB],
       ylim = c(0,8))
  title("Before data only")
  
  # Create plot 2 and add title
  plot(whiteside$Temp, whiteside$Gas,
       ylim = c(0,8))
  title("Complete dataset")
  
  # Create plot 3 and add title
  plot(whiteside$Temp[indexA], whiteside$Gas[indexA],
       ylim = c(0,8))
  title("After data only")
  
  

# 5. Chapter 5: Advanced plot customization and beyond ------
# Iliinsky and Steele's 12 recommended colors
  # Iliinsky and Steele color name vector
  IScolors <- c("red", "green", "yellow", "blue",
                "black", "white", "pink", "cyan",
                "gray", "orange", "brown", "purple")
  
  # Create the data for the barplot
  barWidths <- c(rep(2, 6), rep(1, 6))
  
  # Recreate the horizontal barplot with colored bars
  barplot(rev(barWidths), horiz = TRUE,
          col = rev(IScolors), axes = FALSE,
          names.arg = rev(IScolors), las = 1)


# Using color to enhance a bubbleplot
  # Iliinsky and Steele color name vector
  IScolors <- c("red", "green", "yellow", "blue",
                "black", "white", "pink", "cyan",
                "gray", "orange", "brown", "purple")
  
  # Create the colored bubbleplot
  symbols(Cars93$Horsepower, Cars93$MPG.city,
          circles = Cars93$Cylinders, inches = 0.2, 
          bg = IScolors[as.numeric(Cars93$Cylinders)])


  
# Advanced plot customization and beyond 
  # Create a table of Cylinders by Origin
  tbl <- table(Cars93$Cylinders, Cars93$Origin)
  
  # Create the default stacked barplot
  barplot(tbl)
  
  # Enhance this plot with color
  barplot(tbl, col = IScolors)
  
  
# Saving plot results as files
  # Call png() with the name of the file we want to create
  png("barplot.png")
  
  # Re-create the plot from the last exercise
  barplot(tbl, col = IScolors)
  
  # Save our file and return to our interactive session
  dev.off()
  
  # Verify that we have created the file
  list.files(pattern = "png")
  
  
# The tabplot package and grid graphics
  # Load the insuranceData package
  library(insuranceData)
  
  # Use the data() function to load the dataCar data frame
  data(dataCar)
  
  # Load the tabplot package
  suppressPackageStartupMessages(library(tabplot))
  
  # Generate the default tableplot() display
  tableplot(dataCar)
  
  
# A lattice graphics example
  # Load the lattice package
  library(lattice)
  
  # Use xyplot() to construct the conditional scatterplot
  xyplot(calories ~ sugars | shelf , data = UScereal)
  
  
# A ggplot2 graphics example
  # Load the ggplot2 package
  library(ggplot2)
  
  # Create the basic plot (not displayed): basePlot
  basePlot <- ggplot(Cars93, aes(x = Horsepower, y = MPG.city))
  
  # Display the basic scatterplot
  basePlot + geom_point()
  
  # Color the points by Cylinders value
  basePlot + 
    geom_point(colour = IScolors[Cars93$Cylinders])
  
  # Make the point sizes also vary with Cylinders value
  basePlot + 
    geom_point(colour = IScolors[Cars93$Cylinders], 
               size = as.numeric(Cars93$Cylinders))