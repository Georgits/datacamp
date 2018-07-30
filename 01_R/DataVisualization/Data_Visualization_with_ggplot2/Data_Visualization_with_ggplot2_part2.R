library(ggplot2)
library(ggthemes)
library(dplyr)
library(RColorBrewer)
library(grid)
library(data.table)
library(lattice)

# 1. Chapter 1: ---- 
# Statistics
    
    # Smoothing ----

    # ggplot2 is already loaded
    
    # Explore the mtcars data frame with str()
    str(mtcars)
    
    # A scatter plot with LOESS smooth:
    ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_point() +
      geom_smooth()
    
    # A scatter plot with an ordinary Least Squares linear model:
    ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_point() +
      geom_smooth(method = "lm")
    
    # The previous plot, without CI ribbon:
    ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_point() +
      geom_smooth(method = "lm", se = FALSE)
    
    # The previous plot, without points:
    ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_smooth(method = "lm", se = FALSE)

    
    # Grouping variables ----
    # ggplot2 is already loaded
    
    # Define cyl as a factor variable
    ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
      geom_point() +
      stat_smooth(method = "lm", se = F)
    
    # Complete the following ggplot command as instructed
    ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
      geom_point() +
      stat_smooth(method = "lm", se = F) +
      stat_smooth(method = "lm", se = F, aes(group = 1))
    
    
    
    # Modifying stat_smooth ----
    # Plot 1: change the LOESS span
    ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_point() +
      # Add span below 
      geom_smooth(se = F, span = 0.7)
    
    # Plot 2: Set the overall model to LOESS and use a span of 0.7
    ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
      geom_point() +
      stat_smooth(method = "lm", se = F) +
      # Change method and add span below
      stat_smooth(method = "loess", aes(group = 1), 
                  se = F, col = "black", span = 0.7)
    
    # Plot 3: Set col to "All", inside the aes layer of stat_smooth()
    ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
      geom_point() +
      stat_smooth(method = "lm", se = F) +
      stat_smooth(method = "loess",
                  # Add col inside aes()
                  aes(group = 1, col = "All"), 
                  # Remove the col argument below
                  se = F, span = 0.7)
    
    # Plot 4: Add scale_color_manual to change the colors
    myColors <- c(brewer.pal(3, "Dark2"), "black")
    ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) +
      geom_point() +
      stat_smooth(method = "lm", se = F, span = 0.75) +
      stat_smooth(method = "loess", 
                  aes(group = 1, col="All"), 
                  se = F, span = 0.7) +
      # Add correct arguments to scale_color_manual
      scale_color_manual("Cylinders", values = myColors)
    
    
    # Modifying stat_smooth (2) ----
    # Plot 1: Jittered scatter plot, add a linear model (lm) smooth:
    ggplot(Vocab, aes(x = education, y = vocabulary)) +
      geom_jitter(alpha = 0.2) +
      stat_smooth(method = "lm", se = FALSE)
    
    # Plot 2: Only lm, colored by year
    ggplot(Vocab, aes(x = education, y = vocabulary, col = factor(year))) +
      stat_smooth(method = "lm", se = FALSE)
    
    # Plot 3: Set a color brewer palette
    ggplot(Vocab, aes(x = education, y = vocabulary, col = factor(year))) +
      stat_smooth(method = "lm", se = FALSE) +
      scale_color_brewer()
    
    # Plot 4: Add the group, specify alpha and size
    ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) +
      stat_smooth(method = "lm", se = F, alpha = 0.6, size = 2) +
      scale_color_gradientn(colors = brewer.pal(9,"YlOrRd"))
    
    
    # Quantiles ----
    # Use stat_quantile instead of stat_smooth:
    ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) +
      stat_quantile(alpha = 0.6, size = 2) +
      scale_color_gradientn(colors = brewer.pal(9,"YlOrRd"))
    
    # Set quantile to 0.5:
    ggplot(Vocab, aes(x = education, y = vocabulary, col = year, group = factor(year))) +
      stat_quantile(quantiles = 0.5, alpha = 0.6, size = 2) +
      scale_color_gradientn(colors = brewer.pal(9,"YlOrRd"))
    
    
    # Sum ----
    # Plot with linear and loess model
    p <- ggplot(Vocab, aes(x = education, y = vocabulary)) +
      stat_smooth(method = "loess", aes(col = "red"), se = F) +
      stat_smooth(method = "lm", aes(col = "blue"), se = F) +
      scale_color_discrete("Model", labels = c("red" = "LOESS", "blue" = "lm"))
    
    # Add stat_sum
    # This will map the overall count of each dot onto size. 
    # You don't have to set any arguments, the aesthetics will be inherited from the base plot!
    p + stat_sum()
    
    # Add stat_sum and set size range
    p +  stat_sum() + scale_size(range = c(1,10))
    
    
    
    
    # Stats outside Geoms: Preparations ----
    # Display structure of mtcars
    str(mtcars)
    
    # Convert cyl and am to factors:
    mtcars$cyl <- as.factor(mtcars$cyl)
    mtcars$am <- as.factor(mtcars$am)
    
    # Define positions:
    posn.d <- position_dodge(width = 0.1)
    posn.jd <- position_jitterdodge(jitter.width = 0.1, dodge.width = 0.2) 
    posn.j <- position_jitter(width = 0.2)
    
    # base layers:
    wt.cyl.am <- ggplot(mtcars, aes(x = cyl, y = wt, col = am, fill = am, group = am))
    
    
    # Plotting variations ----
    # Remember that you can always specify your own function to the fun.data argument as long 
    # as the variable names match the aesthetics that you will need for the geom layer. 
    
    # Plot 1: Jittered, dodged scatter plot with transparent points
    wt.cyl.am +
      geom_point(position = posn.jd, alpha = 0.6)
    
    # Plot 2: Mean and SD - the easy way
    wt.cyl.am +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), position = posn.d)
    
    # Plot 3: Mean and 95% CI - the easy way
    wt.cyl.am +
      stat_summary(fun.data = mean_cl_normal,position = posn.d)
    
    # Plot 4: Mean and SD - with T-tipped error bars - fill in ___
    wt.cyl.am +
      stat_summary(geom = "point", fun.y = mean, 
                   position = posn.d) +
      stat_summary(geom = "errorbar", fun.data = mean_sdl, 
                   position = posn.d, fun.args = list(mult = 1), width = 0.1)
    
    
    
    
    # Custom Functions ----
    xx <- seq(1,100,1)
    
    # Play vector xx is available
    
    # Function to save range for use in ggplot 
    gg_range <- function(x) {
      # Change x below to return the instructed values
      data.frame(ymin = min(x), # Min
                 ymax = max(x)) # Max
    }
    
    gg_range(xx)
    # Required output:
    #   ymin ymax
    # 1    1  100
    
    # Function to Custom function:
    med_IQR <- function(x) {
      # Change x below to return the instructed values
      data.frame(y = median(x), # Median
                 ymin = quantile(x)[2], # 1st quartile
                 ymax = quantile(x)[4])  # 3rd quartile
    }
    
    med_IQR(xx)
    # Required output:
    #        y  ymin  ymax
    # 25% 50.5 25.75 75.25
    
    
    # Custom Functions (2) ----
    # The base ggplot command, you don't have to change this
    wt.cyl.am <- ggplot(mtcars, aes(x = cyl,y = wt, col = am, fill = am, group = am))
    
    # Add three stat_summary calls to wt.cyl.am
    wt.cyl.am + 
      stat_summary(geom = "linerange", fun.data = med_IQR, 
                   position = posn.d, size = 3) +
      stat_summary(geom = "linerange", fun.data = gg_range, 
                   position = posn.d, size = 3, 
                   alpha = 0.4) +
      stat_summary(geom = "point", fun.y = median, 
                   position = posn.d, size = 3, 
                   col = "black", shape = "X")

    
    
# 2. Chapter 2: ---- 
# Coordinates and Facets ----
    # Zooming In ----
    # Basic ggplot() command, coded for you
    p <- ggplot(mtcars, aes(x = wt, y = hp, col = am)) + geom_point() + geom_smooth()
    
    # Add scale_x_continuous
    p + scale_x_continuous(limits = c(3, 6), expand = c(0,0))
    
    # The proper way to zoom in:
    p + coord_cartesian(xlim = c(3,6))
    
    
    
    # Aspect Ratio ----
    # Complete basic scatter plot function
    base.plot <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, col = Species)) +
      geom_jitter() +
      geom_smooth(method = "lm", se = F)
    
    # Plot base.plot: default aspect ratio
    base.plot
    
    # Fix aspect ratio (1:1) of base.plot
    base.plot + coord_equal(1:1)
    
    
    
    # Pie Charts ----
    # Create stacked bar plot: thin.bar
    thin.bar <- ggplot(mtcars, aes(x = 1, fill = cyl)) +
      geom_bar()
    
    # Convert thin.bar to pie chart
    # Set the argument theta to "y". This specified the axis which would be transformed to polar coordinates. 
    # There's a ring structure instead of a pie!
    thin.bar + coord_polar(theta = "y")
    
    # Create stacked bar plot: wide.bar
    wide.bar <- ggplot(mtcars, aes(x = 1, fill = cyl)) +
      geom_bar(width = 1)
    
    # Convert wide.bar to pie chart
    wide.bar + coord_polar(theta = "y")
    
    
    
    # Facets: the basics ----
    # Basic scatter plot:
    p <- ggplot(mtcars, aes(x = wt, y = mpg)) +
      geom_point()
    
    # Separate rows according to transmission type, am
    p + facet_grid(~ am )
    
    # Separate columns according to cylinders, cyl
    p + facet_grid(~ cyl )
    
    # Separate by both columns and rows 
    p + facet_grid(am ~ cyl)
    
    
    
    
    # !!!! Many variables ----
    # Code to create the cyl_am col and myCol vector
    mtcars$cyl_am <- paste(mtcars$cyl, mtcars$am, sep = "_")
    myCol <- rbind(brewer.pal(9, "Blues")[c(3,6,8)],
                   brewer.pal(9, "Reds")[c(3,6,8)])
    
    # Basic scatter plot, add color scale:
    ggplot(mtcars, aes(x = wt, y = mpg, col = cyl_am)) +
      geom_point() +
      scale_color_manual(values = myCol)
    
    # Facet according on rows and columns.
    ggplot(mtcars, aes(x = wt, y = mpg, col = cyl_am)) +
      geom_point() +
      scale_color_manual(values = myCol) +
      facet_grid(gear ~ vs)
    
    # Add more variables
    ggplot(mtcars, aes(x = wt, y = mpg, col = cyl_am, size = disp)) + geom_point() +
      scale_color_manual(values = myCol) +
      facet_grid(gear ~ vs)
    
    
    # Dropping levels ----
    # When you have a categorical variable with many levels which are not all present in sub-groups of another variable, 
    # it may be desirable to drop the unused levels. 
    # Basic scatter plot
    ggplot(msleep, aes(x = sleep_total, y = name, col = order)) +
      geom_point()
    
    # Facet rows accoding to vore
    ggplot(msleep, aes(x = sleep_total, y = name, col = order)) +
      geom_point() +
      facet_grid(vore ~ .)
    
    # !!! Specify scale and space arguments to free up rows
    ggplot(msleep, aes(x = sleep_total, y = name, col = order)) +
      geom_point() +
      facet_grid(vore ~ ., scale = "free_y", space = "free_y")
    
    
    
    
    
# 3. Chapter 3: Themes ---- 
# Themes from Scratch
    z <- ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + 
      geom_point() + 
      facet_grid(~ cyl) + 
      geom_smooth(method = "lm", se = FALSE)
    
    myPink <- "#FEE0D2"
    myRed <-  "#99000D"
    
    # Rectangles
    # Plot 1: change the plot background color to myPink:
    z + theme(plot.background = element_rect(fill = myPink))
    
    # Plot 2: adjust the border to be a black line of size 3
    z + theme(plot.background = element_rect(fill = myPink, color = "black", size = 3))
    
    # Plot 3: set panel.background, legend.key, legend.background and strip.background to element_blank()
    uniform_panels <- theme(panel.background = element_blank(), 
                            legend.key = element_blank(), 
                            legend.background = element_blank(), 
                            strip.background = element_blank())
    z + theme(plot.background = element_rect(fill = myPink, color = "black", size = 3)) + uniform_panels
    
    
    # Lines ----
    # Extend z with theme() function and three arguments
    z + theme(panel.grid = element_blank(),
              axis.line = element_line(color = "black"),
              axis.ticks = element_line(color = "black"))
    
    
    # Text ----
    # Extend z with theme() function and four arguments
    z + theme(
      strip.text = element_text(size = 16, color = myRed),
      axis.title.y = element_text(color = myRed, hjust = 0, face = "italic"),
      axis.title.x = element_text(color = myRed, hjust = 0, face = "italic"),
      axis.text = element_text(color = "black"))
    
    
    
    # Legends ----
    # Move legend by position
    z + theme(legend.position = c(0.85, 0.85))
    
    # Change direction
    z + theme(legend.direction = "horizontal")
    
    # Change location by name
    z + theme(legend.position = "bottom")
    
    # Remove legend entirely
    z + theme(legend.position = "none")
    
    
    # positions ----
    library(grid)
    # Increase spacing between facets
    z + theme(panel.spacing.x = unit(2, "cm"))
    
    
    # Add code to remove any excess plot margin space
    z + theme(panel.spacing.x = unit(2, "cm"),
              plot.margin = unit(c(0,0,0,0), "cm"))
    
    
    
    
    # Update Themestheme update----
    # Theme layer saved as an object, theme_pink
    theme_pink <- theme(panel.background = element_blank(),
                        legend.key = element_blank(),
                        legend.background = element_blank(),
                        strip.background = element_blank(),
                        plot.background = element_rect(fill = myPink, color = "black", size = 3),
                        panel.grid = element_blank(),
                        axis.line = element_line(color = "black"),
                        axis.ticks = element_line(color = "black"),
                        strip.text = element_text(size = 16, color = myRed),
                        axis.title.y = element_text(color = myRed, hjust = 0, face = "italic"),
                        axis.title.x = element_text(color = myRed, hjust = 0, face = "italic"),
                        axis.text = element_text(color = "black"),
                        legend.position = "none")
    
    # Apply theme_pink to z2
    z + theme_pink
    
    # Change code so that old theme is saved as old
    old <- theme_update(panel.background = element_blank(),
                        legend.key = element_blank(),
                        legend.background = element_blank(),
                        strip.background = element_blank(),
                        plot.background = element_rect(fill = myPink, color = "black", size = 3),
                        panel.grid = element_blank(),
                        axis.line = element_line(color = "black"),
                        axis.ticks = element_line(color = "black"),
                        strip.text = element_text(size = 16, color = myRed),
                        axis.title.y = element_text(color = myRed, hjust = 0, face = "italic"),
                        axis.title.x = element_text(color = myRed, hjust = 0, face = "italic"),
                        axis.text = element_text(color = "black"),
                        legend.position = "none")
    
    # Display the plot z2
    z
    
    # Restore the old plot
    theme_set(old)
    
    
    
    # Exploring ggthemes ----
    # Apply theme_tufte
    z + theme_tufte()
    
    # Apply theme_tufte, modified:
    z + theme_tufte() +
      theme(
        legend.position = c(0.9, 0.9),
        legend.title = element_text(face = "italic", size = 12),
        axis.title = element_text(face = "bold", size = 14)
      )
    
    

# 4. Chapter 4: ---- 
# Best Practices: Bar Plots ----
    # Bar Plots (1) ----
    # Base layers
    m <- ggplot(mtcars, aes(x = cyl, y = wt))
    
    # Draw dynamite plot
    m +
      stat_summary(fun.y = mean, geom = "bar", fill = "skyblue") +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.1)
    
    
    # Bar Plots (2) ----
    # Base layers
    m <- ggplot(mtcars, aes(x = cyl,y = wt, col = am, fill = am))
    
    # Plot 1: Draw dynamite plot
    m +
      stat_summary(fun.y = mean, geom = "bar") +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.1)
    
    # Plot 2: Set position dodge in each stat function
    m +
      stat_summary(fun.y = mean, geom = "bar", position = "dodge") +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), 
                   geom = "errorbar", width = 0.1, position = "dodge")
    
    # Set your dodge posn manually
    posn.d <- position_dodge(0.9)
    
    # Plot 3:  Redraw dynamite plot
    m +
      stat_summary(fun.y = mean, geom = "bar", position = posn.d) +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "errorbar", width = 0.1, position = posn.d)
    
    
    
    
    
    # Bar Plots (3) ----
    mtcars.cyl <- data.table(mtcars)[,list(wt.avg = mean(wt), sd = sd(wt), n = .N, prop = (.N/nrow(mtcars))), by = "cyl"]
    
    # Base layers
    m <- ggplot(mtcars.cyl, aes(x = cyl, y = wt.avg))
    
    # Plot 1: Draw bar plot
    m + geom_bar(stat = "identity", fill = "skyblue")
    
    # Plot 2: Add width aesthetic
    m + geom_bar(stat = "identity", aes(width = prop), fill = "skyblue")
    
    # Plot 3: Add error bars
    m + geom_bar(stat = "identity", aes(width = prop), fill = "skyblue") +
      geom_errorbar(aes(ymin = wt.avg - sd, ymax = wt.avg + sd), width = 0.1)
  
    
    
# Best Practices: Pie Charts ----
    # Pie Charts (1) ----
    # Convert bar chart to pie chart
    # This is already pretty good, but there's a small hole in the center of the pies.
    # Inside geom_bar() set width = 1 so that the bars fill up the entire width resulting in a full pie chart.
    ggplot(mtcars, aes(x = factor(1), fill = am)) +
      geom_bar(position = "fill", width = 1) +
      facet_grid(~ cyl) + 
      coord_polar(theta = "y")
    
    
    # Pie Charts (2) ----
    # Parallel coordinates plot using GGally
    library(GGally)
    
    # All columns except am
    group_by_am <- 9
    my_names_am <- (1:11)[-group_by_am]
    
    # Basic parallel plot - each variable plotted as a z-score transformation
    ggparcoord(mtcars, my_names_am, groupColumn = group_by_am, alpha = 0.8)
    
    
    
    # !!!! Plot Matrix (1) and (2) ----
    mtcars %>% 
      select(mpg, disp, drat, wt, qsec) %>% 
      GGally::ggpairs()
    
    
    
    # Heat Maps ----
    # Create color palette
    myColors <- brewer.pal(9, "Reds")
    
    # Build the heat map from scratch
    ggplot(barley, aes(x = year, y = variety, fill = yield)) +
      geom_tile() +
      facet_wrap(~ site, ncol = 1) +
      scale_fill_gradientn(colors = myColors)
    

    
    # Heat Maps Alternatives (1) ----
    # The heat map we want to replace
    # Don't remove, it's here to help you!
    myColors <- brewer.pal(9, "Reds")
    ggplot(barley, aes(x = year, y = variety, fill = yield)) +
      geom_tile() +
      facet_wrap( ~ site, ncol = 1) +
      scale_fill_gradientn(colors = myColors)
    
    # Line plots
    ggplot(barley, aes(x = year, y = yield, col = variety, group = variety)) +
      geom_line() + 
      facet_wrap(~ site, nrow = 1)
    
    
    # Heat Maps Alternatives (2) ----
    # Create overlapping ribbon plot from scratch
    ggplot(barley, aes(x = year, y = yield, col = site , group = site, fill = site)) +
      stat_summary(fun.y = mean, geom = "line") +
      stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "ribbon", col = NA, alpha = 0.1)
    
    
    
# 5. Chapter 5: ---- 
# Case Study ----
    library(foreign)
    pathData <- file.path(getwd(), "chis12_adult_stata", "Data")
    
    adult <- read.dta(file.path(pathData, "ADULT.dta")) 
    # Explore the dataset with summary and str
    summary(adult)
    str(adult)
    
    # Age histogram
    ggplot(adult, aes(x = srage_p1)) + geom_histogram(stat = "count")
    
    # BMI histogram
    ggplot(adult, aes(x = bmi_p)) + geom_histogram()
    
    # Age colored by BMI, default binwidth
    ggplot(adult, aes(x = srage_p1, fill = factor(rbmi))) + geom_histogram(binwidth = 1, stat = "count")
    
    
    # Data Cleaning ----
    # # Remove individual aboves 84
    # adult <- adult[adult$srage_p1 <= 84, ] 
    # 
    # # Remove individuals with a BMI below 16 and above or equal to 52
    # adult <- adult[adult$bmi_p >= 16 & adult$bmi_p < 52, ]
    # 
    # # Relabel the race variable:
    # adult$RACEHPR2 <- factor(adult$RACEHPR2, labels = c("Latino", "Asian", "African American", "White"))
    # 
    # # Relabel the BMI categories variable:
    # adult$rbmi <- factor(adult$rbmi, labels = c("Under-weight", "Normal-weight", "Over-weight", "Obese"))
    # 
    
    # Multiple Histograms ----
    # The dataset adult is available
    
    # The color scale used in the plot
    BMI_fill <- scale_fill_brewer("BMI Category", palette = "Reds")
    
    # Theme to fix category display in faceted plot
    fix_strips <- theme(strip.text.y = element_text(angle = 0, hjust = 0, vjust = 0.1, size = 14),
                        strip.background = element_blank(), 
                        legend.position = "none")
    
    # Histogram, add BMI_fill and customizations
    ggplot(adult, aes(x = srage_p1, fill =  factor(rbmi))) + 
      geom_histogram(binwidth = 1, stat = "count") +
      fix_strips +
      BMI_fill +
      facet_grid(~ rbmi) +
      theme_classic()
    
    
    # Alternatives ----
    # Plot 1 - Count histogram
    ggplot(adult, aes (x = srage_p1, fill = factor(rbmi))) + 
      geom_histogram(binwidth = 1, stat = "count") +
      BMI_fill
    
    # # Plot 2 - Density histogram
    # ggplot(adult, aes (x = srage_p1, fill = factor(rbmi))) + 
    #   geom_histogram(aes(y = ..density..), binwidth = 1, stat = "count") +
    #   BMI_fill
    
    # Plot 3 - Faceted count histogram
    ggplot(adult, aes (x = srage_p1, fill= factor(rbmi))) + 
      geom_histogram(binwidth = 1, stat = "count") +
      BMI_fill +
      facet_grid(rbmi ~.)
    
    # # Plot 4 - Faceted density histogram
    # ggplot(adult, aes (x = srage_p1, fill= factor(rbmi))) + 
    #   geom_histogram(aes(y = ..density..), binwidth = 1) +
    #   BMI_fill + 
    #   facet_grid(rbmi ~ .)
    
    # # Plot 5 - Density histogram with position = ill"
    # ggplot(adult, aes (x = srage_p1, fill= factor(rbmi))) + 
    #   geom_histogram(aes(y = ..density..), binwidth = 1, position = "fill", stat = "count") +
    #   BMI_fill
    
    # Plot 6 - The accurate histogram
    ggplot(adult, aes (x = srage_p1, fill= factor(rbmi))) + 
      geom_histogram(aes(y = ..count../sum(..count..)), binwidth = 1, position = "fill", stat = "count") +
      BMI_fill
    
    
    
    # Do Things Manually -----
    # An attempt to facet the accurate frequency rbmi from before (failed)
    ggplot(adult, aes (x = srage_p1, fill= factor(rbmi))) + 
      geom_histogram(aes(y = ..count../sum(..count..)), binwidth = 1, position = "fill", stat = "count") +
      BMI_fill +
      facet_grid(rbmi ~ .)
    
    # Create DF with table()
    DF <- table(adult$rbmi, adult$srage_p1)
    
    # Use apply on DF to get frequency of each group
    DF_freq <- apply(DF, 2, function(x) x/sum(x))
    
    # Load reshape2 and use melt on DF to create DF_melted
    library(reshape2)
    
    # Change names of DF_melted
    DF_melted <- melt(DF_freq)
    names(DF_melted) <- c("FILL", "X", "value")
    
    # Add code to make this a faceted plot
    ggplot(DF_melted, aes(x = X, y = value, fill = FILL)) +
      geom_bar(stat = "identity", position = "stack") +
      BMI_fill + 
      facet_grid(FILL ~ .)
    
    
    
    # Merimeko/Mosaic Plot ----
    # The initial contingency table
    DF <- as.data.frame.matrix(table(adult$srage_p1, adult$rbmi))
    
    # Add the columns groupsSum, xmax and xmin. Remove groupSum again.
    DF$groupSum <- rowSums(DF)
    DF$xmax <- cumsum(DF$groupSum)
    DF$xmin <- DF$xmax - DF$groupSum
    # The groupSum column needs to be removed, don't remove this line
    DF$groupSum <- NULL
    
    # Copy row names to variable X
    DF$X <- row.names(DF)
    
    # Melt the dataset
    library(reshape2)
    DF_melted <- melt(DF, id.vars = c("X", "xmin", "xmax"), variable.name = "FILL")
    
    # dplyr call to calculate ymin and ymax - don't change
    library(dplyr)
    DF_melted <- DF_melted %>% 
      group_by(X) %>% 
      mutate(ymax = cumsum(value/sum(value)),
             ymin = ymax - value/sum(value))
    
    # Plot rectangles - don't change.
    library(ggthemes)
    ggplot(DF_melted, aes(ymin = ymin, 
                          ymax = ymax,
                          xmin = xmin, 
                          xmax = xmax, 
                          fill = FILL)) + 
      geom_rect(colour = "white") +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0)) +
      BMI_fill +
      theme_tufte()
    
    
    
    # Adding statistics ----
    # Perform chi.sq test (RBMI and SRAGE_P)
    results <- chisq.test(table(adult$rbmi, adult$srage_p1))
    
    # Melt results$residuals and store as resid
    resid <- melt(results$residuals)
    
    # Change names of resid
    names(resid) <- c("FILL", "X", "residual")
    
    # merge the two datasets:
    DF_all <- merge(DF_melted, resid, by = c("X", "FILL"))
    
    # Update plot command
    library(ggthemes)
    ggplot(DF_all, aes(ymin = ymin, 
                       ymax = ymax,
                       xmin = xmin, 
                       xmax = xmax, 
                       fill = residual)) + 
      geom_rect() +
      scale_fill_gradient2() +
      scale_x_continuous(expand = c(0,0)) +
      scale_y_continuous(expand = c(0,0)) +
      theme_tufte()
    
    
    # Adding text ----
    # Position for labels on x axis
    DF_all$xtext <- DF_all$xmin + (DF_all$xmax - DF_all$xmin)/2
    
    # Position for labels on y axis (don't change)
    index <- DF_all$xmax == max(DF_all$xmax)
    DF_all$ytext <- DF_all$ymin[index] + (DF_all$ymax[index] - DF_all$ymin[index])/2
    
    # Plot
    ggplot(DF_all, aes(ymin = ymin, ymax = ymax, xmin = xmin, 
                       xmax = xmax, fill = residual)) + 
      geom_rect(col = "white") +
      # geom_text for ages (i.e. the x axis)
      geom_text(aes(x = xtext, 
                    label = X),
                y = 1,
                size = 3,
                angle = 90,
                hjust = 1,
                show.legend = FALSE) +
      # geom_text for BMI (i.e. the fill axis)
      geom_text(aes(x = max(xmax), 
                    y = ytext,
                    label = FILL),
                size = 3,
                hjust = 1,
                show.legend  = FALSE) +
      scale_fill_gradient2() +
      theme_tufte() +
      theme(legend.position = "bottom")
    
    
    
    # Generalizations ----
    # Load all packages
    library(ggplot2)
    library(reshape2)
    library(dplyr)
    library(ggthemes)
    
    # Script generalized into a function
    mosaicGG <- function(data, X, FILL) {
      
      # Proportions in raw data
      DF <- as.data.frame.matrix(table(data[[X]], data[[FILL]]))
      DF$groupSum <- rowSums(DF)
      DF$xmax <- cumsum(DF$groupSum)
      DF$xmin <- DF$xmax - DF$groupSum
      DF$X <- row.names(DF)
      DF$groupSum <- NULL
      DF_melted <- melt(DF, id = c("X", "xmin", "xmax"), variable.name = "FILL")
      library(dplyr)
      DF_melted <- DF_melted %>% 
        group_by(X) %>% 
        mutate(ymax = cumsum(value/sum(value)),
               ymin = ymax - value/sum(value))
      
      # Chi-sq test
      results <- chisq.test(table(data[[FILL]], data[[X]])) # fill and then x
      resid <- melt(results$residuals)
      names(resid) <- c("FILL", "X", "residual")
      
      # Merge data
      DF_all <- merge(DF_melted, resid)
      
      # Positions for labels
      DF_all$xtext <- DF_all$xmin + (DF_all$xmax - DF_all$xmin)/2
      index <- DF_all$xmax == max(DF_all$xmax)
      DF_all$ytext <- DF_all$ymin[index] + (DF_all$ymax[index] - DF_all$ymin[index])/2
      
      # plot:
      g <- ggplot(DF_all, aes(ymin = ymin,  ymax = ymax, xmin = xmin, 
                              xmax = xmax, fill = residual)) + 
        geom_rect(col = "white") +
        geom_text(aes(x = xtext, label = X),
                  y = 1, size = 3, angle = 90, hjust = 1, show.legend = FALSE) +
        geom_text(aes(x = max(xmax),  y = ytext, label = FILL),
                  size = 3, hjust = 1, show.legend = FALSE) +
        scale_fill_gradient2("Residuals") +
        scale_x_continuous("Individuals", expand = c(0,0)) +
        scale_y_continuous("Proportion", expand = c(0,0)) +
        theme_tufte() +
        theme(legend.position = "bottom")
      print(g)
    }
    
    # BMI described by age
    mosaicGG(adult, "srage_p1","rbmi")
    
    # Poverty described by age
    mosaicGG(adult, "srage_p1","povll")
    
    # mtcars: am described by cyl
    mosaicGG(mtcars, "cyl","am")
    
    # Vocab: vocabulary described by education
    library(car)
    mosaicGG(Vocab, "education","vocabulary")