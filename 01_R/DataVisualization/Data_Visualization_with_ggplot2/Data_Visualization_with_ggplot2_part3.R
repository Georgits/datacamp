# Create movies_small
library(ggplot2)
library(ggplot2movies)
library(tidyr)
library(MASS)
library(dplyr)
library(datasets)
library(viridis)
library(ggplot2)
library(reshape2)
library(ggtern)
library(geomnet)
library(ggfortify)
library(ggmap)
library(ggthemes)
library(rgdal)
library(gganimate)
library(car)
library(grid)
library(gtable)
library(aplpack)

set.seed(123)
movies_small <- movies[sample(nrow(movies), 1000), ]
movies_small$rating <- factor(round(movies_small$rating))


# 1. Chapter 1: ---- 
# Statistical plots

  # Refresher (1) ----
  # Explore movies_small with str()
  str(movies_small)
  
  # Build a scatter plot with mean and 95% CI
  ggplot(movies_small, aes(x = rating, y = votes)) +
    geom_point() +
    stat_summary(fun.data = "mean_cl_normal",
                 geom = "crossbar",
                 width = 0.2,
                 col = "red") +
    scale_y_log10()
  
  # Refresher (2) ----
  # Reproduce the plot
  ggplot(diamonds, aes(x = carat, y = price, col = color)) +
    geom_point(alpha = 0.5, size = 0.5, shape = 16) +
    scale_x_log10(expression(log[10](Carat)), limits = c(0.1,10)) +
    scale_y_log10(expression(log[10](Price)), limits = c(100,100000)) +
    scale_color_brewer(palette = "YlOrRd") +
    coord_equal(1:1) + 
    theme_classic()
  
  
  
  # Refresher (3) ----
  # Add smooth layer and facet the plot
  ggplot(diamonds, aes(x = carat, y = price, col = color)) +
    geom_point(alpha = 0.5, size = .5, shape = 16) +
    scale_x_log10(expression(log[10](Carat)), limits = c(0.1,10)) +
    scale_y_log10(expression(log[10](Price)), limits = c(100,100000)) +
    scale_color_brewer(palette = "YlOrRd") +
    coord_equal() +
    theme_classic() +
    stat_smooth(method = "lm") +
    facet_grid(. ~ cut)
  
  
  
  
  
  # Box Plots -----
  # Transformations
  # movies_small is available
  
  # Add a boxplot geom
  d <- ggplot(movies_small, aes(x = rating, y = votes)) +
    geom_point() +
    geom_boxplot() +
    stat_summary(fun.data = "mean_cl_normal",
                 geom = "crossbar",
                 width = 0.2,
                 col = "red")
  
  # Untransformed plot
  d
  
  # Transform the scale
  d + scale_y_log10()
  
  # Transform the coordinates
  d + coord_trans(y = "log10")
  
  
  # !!!!Cut it up! ----
  # Plot object p
  p <- ggplot(diamonds, aes(x = carat, y = price))
  
  # Use cut_interval
  p + geom_boxplot(aes(group = cut_interval(carat, n = 10)))
  
  # Use cut_number
  p + geom_boxplot(aes(group = cut_number(carat, n = 10)))
  
  # Use cut_width
  p + geom_boxplot(aes(group = cut_width(carat, width = 0.25)))
  
  
  
  # geom_density() ----
  # bimodal Distribution
  mu1 <- log(1)
  mu2 <- log(50)
  sig1 <- log(3)
  sig2 <- log(3)
  cpct <- 0.4
  bimodalDistFunc <- function (n,cpct, mu1, mu2, sig1, sig2) {
  y0 <- rlnorm(n,mean=mu1, sd = sig1)
  y1 <- rlnorm(n,mean=mu2, sd = sig2)
  flag <- rbinom(n,size=1,prob=cpct)
  y <- y0*(1 - flag) + y1*flag
  }
  bimodalData <- bimodalDistFunc(n=200,cpct,mu1,mu2, sig1,sig2)

  test_data <- data.frame(norm = rnorm(200), bimodal = bimodalData, uniform = runif(200))
  # test_data is available
  
  # Calculating density: d
  d <- density(test_data$norm)
  
  # Use which.max() to calculate mode
  mode <- d$x[which.max(d$y)]
  
  # Finish the ggplot call
  ggplot(test_data, aes(x = norm)) +
    geom_rug() +
    geom_density() +
    geom_vline(xintercept = mode, col = "red")
  
  

  
  # Combine density plots and histogram ----
  # test_data is available
  
  # Arguments you'll need later on
  fun_args <- list(mean = mean(test_data$norm), sd = sd(test_data$norm))
  
  # Finish the ggplot
  ggplot(test_data, aes(x = norm)) +
    geom_histogram(aes(y = ..density..)) +
    geom_density(col = "red") +
    stat_function(fun = dnorm, args = fun_args, col = "blue")
  
  
  
  # Adjusting density plots ----
  # small_data is available
  
  # Get the bandwith
  x <- c(-3.5, 0.0, 0.5, 6.0)
  small_data <- as.data.frame(x)
  get_bw <- density(small_data$x)$bw
  
  # Basic plotting object
  p <- ggplot(small_data, aes(x = x)) +
    geom_rug() +
    coord_cartesian(ylim = c(0,0.5))
  
  # Create three plots
  p + geom_density()
  p + geom_density(adjust = 0.25)
  p + geom_density(bw = 0.25 * get_bw)
  
  # Create two plots
  p + geom_density(kernel = "r")
  p + geom_density(kernel = "e")
  
  
  
  # Box plots with varying width ----
  # Finish the plot
  ggplot(diamonds, aes(x = cut, y = price)) +
    geom_boxplot(varwidth = TRUE, aes(col = color)) +
    facet_grid(. ~ color)
  
  
  # Mulitple density plots ----
  test_data2 <- gather(test_data)
  # test_data and test_data2 are available
  str(test_data)
  str(test_data2)
  
  # Plot with test_data
  ggplot(test_data, aes(x = norm)) +
    geom_rug() +
    geom_density()
  
  # Plot two distributions with test_data2
  ggplot(test_data2, aes(x = value, fill = dist, col = dist)) +
    geom_rug(alpha = 0.6) +
    geom_density(alpha = 0.6)
  
  
  
  # Multiple density plots (2) ----
  # Individual densities
  ggplot(msleep[msleep$vore == "herbi", ], aes(x = sleep_total, fill = vore)) +
    geom_density(col = NA, alpha = 0.35) +
    scale_x_continuous(limits = c(0, 24)) +
    coord_cartesian(ylim = c(0, 0.3))
  
  # With faceting
  ggplot(msleep, aes(x = sleep_total, fill = vore)) +
    geom_density(col = NA, alpha = 0.35) +
    scale_x_continuous(limits = c(0, 24)) +
    coord_cartesian(ylim = c(0, 0.3)) +
    facet_wrap(~ vore, nrow = 2)
  
  # Note that by default, the x ranges fill the scale
  ggplot(msleep, aes(x = sleep_total, fill = vore)) +
    geom_density(col = NA, alpha = 0.35) +
    scale_x_continuous(limits = c(0, 24)) +
    coord_cartesian(ylim = c(0, 0.3))
  
  # Trim each density plot individually
  ggplot(msleep, aes(x = sleep_total, fill = vore)) +
    geom_density(col = NA, alpha = 0.35, trim = TRUE) +
    scale_x_continuous(limits=c(0,24)) +
    coord_cartesian(ylim = c(0, 0.3))
  
  
  
  
  
  # Non-weighted density plots ----
  # Density plot from before
  ggplot(msleep, aes(x = sleep_total, fill = vore)) +
    geom_density(col = NA, alpha = 0.35) +
    scale_x_continuous(limits = c(0, 24)) +
    coord_cartesian(ylim = c(0, 0.3))
  
  # Finish the dplyr command
  msleep2 <- msleep %>%
    group_by(vore) %>%
    mutate(n = n() / nrow(msleep), test = n())
  
  # Density plot, weighted
  ggplot(msleep2, aes(x = sleep_total, fill = vore)) +
    geom_density(aes(weight = n), col = NA, alpha = 0.35) +
    scale_x_continuous(limits = c(0, 24)) +
    coord_cartesian(ylim = c(0, 0.3))
  
  # Violin plot
  ggplot(msleep, aes(x = vore, y = sleep_total, fill = vore)) +
    geom_violin()
  
  # Violin plot, weighted
  ggplot(msleep2, aes(x = vore, y = sleep_total, fill = vore)) +
    geom_violin(aes(weight = n), col = NA)
  
  
  
  
  
  # 2D density plots (1) ----
  # Base layers
  p <- ggplot(faithful, aes(x = waiting, y = eruptions)) +
    scale_y_continuous(limits = c(1, 5.5), expand = c(0, 0)) +
    scale_x_continuous(limits = c(40, 100), expand = c(0, 0)) +
    coord_fixed(60 / 4.5)
  
  # Use geom_density_2d()
  p + geom_density_2d()
  
  # Use stat_density_2d()
  p + stat_density_2d(h = c(5, 0.5), aes(col = ..level..))
  
  
  
  
  # 2D density plots (2) ----
  # Add viridis color scale
  ggplot(faithful, aes(x = waiting, y = eruptions)) +
    scale_y_continuous(limits = c(1, 5.5), expand = c(0,0)) +
    scale_x_continuous(limits = c(40, 100), expand = c(0,0)) +
    coord_fixed(60/4.5) +
    stat_density_2d(geom = "tile", aes(fill = ..density..), h=c(5,.5), contour = FALSE) +
    scale_fill_viridis()
  
  
# 2. Chapter 2: Plots for specific data types (Part 1)  ---- 
# Plots for specific data types 
  
  # Pair plots and correlation matrices ----
  # pairs
  pairs(iris[1:4])
  
  # chart.Correlation
  library(PerformanceAnalytics)
  chart.Correlation(iris[1:4])
  
  # ggpairs
  library(GGally)
  ggpairs(mtcars[1:3])
  
  
  
  
  
  # Create a correlation matrix in ggplot2 ---
  
  cor_list <- function(x) {
    L <- M <- cor(x)
    
    M[lower.tri(M, diag = TRUE)] <- NA
    M <- melt(M)
    names(M)[3] <- "points"
    
    L[upper.tri(L, diag = TRUE)] <- NA
    L <- melt(L)
    names(L)[3] <- "labels"
    
    merge(M, L)
  }
  
  # Calculate xx with cor_list
  library(dplyr)
  xx <- iris %>%
    group_by(Species) %>%
    do(cor_list(.[1:4])) 
  
  # Finish the plot
  ggplot(xx, aes(x = Var1, y = Var2)) +
    geom_point(aes(col = points, size = abs(points)), shape = 16) +
    geom_text(aes(col = labels,  size = abs(labels)), label = round(labels, 2)) +
    scale_size(range = c(0, 6)) +
    scale_color_gradient("r", limits = c(-1, 1)) +
    scale_y_discrete("", limits = rev(levels(xx$Var1))) +
    scale_x_discrete("") +
    guides(size = FALSE) +
    geom_abline(slope = -1, intercept = nlevels(xx$Var1) + 1) +
    coord_fixed() +
    facet_grid(. ~ Species) +
    theme(axis.text.y = element_text(angle = 45, hjust = 1),
          axis.text.x = element_text(angle = 45, hjust = 1),
          strip.background = element_blank())
  
  
  
  # Proportional/stacked bar plots ----
  # Explore africa
  library(GSIF)
  data(afsp)
  africa <- afsp$horizons %>% 
              rename(Sand = SNDPPT, Silt = SLTPPT, Clay = CLYPPT) %>% 
              select(Sand, Silt, Clay)
  africa <- africa[complete.cases(africa),]
  str(africa)
  
  # Sample the dataset
  africa_sample <- africa[sample(1:nrow(africa), size = 50), ]
  
  # Add an ID column from the row.names
  africa_sample$ID <- row.names(africa_sample)
  
  # Gather africa_sample
  library(tidyr)
  africa_sample_tidy <- gather(africa_sample, key, value, -ID)
  head(africa_sample_tidy)
  
  # Finish the ggplot command
  ggplot(africa_sample_tidy, aes(x = factor(ID), y = value, fill = key)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = "Location", y = "Composition", fill = "Component") +
    theme_minimal()
  
  
  
  # Producing ternary plots ----
  # Build ternary plot
  ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
    geom_point(shape = 16, alpha = 0.2)
  
  
  # !!! Adjusting ternary plots ----
  # ggtern and ggplot2 are loaded
  
  # Plot 1
  ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
    geom_density_tern(alpha = 0.5)
  
  # Plot 2
  ggtern(africa, aes(x = Sand, y = Silt, z = Clay)) +
    stat_density_tern(aes(fill = ..level.., alpha = ..level..), geom = "polygon", alpha = 0.5) + 
    guides(fill = guide_legend(show = FALSE))
  
  
  # Build the network (1) ----
  # Examine structure of madmen
  str(geomnet::madmen)
  
  # Merge edges and vertices
  mmnet <- merge(madmen$edges, madmen$vertices,
                 by.x = "Name1", by.y = "label",
                 all = TRUE)
  
  # Examine structure of mmnet
  str(mmnet)
  
  
  
  # Build the network (2) -----
  # Finish the ggplot command
  ggplot(data = mmnet, aes(from_id = Name1, to_id = Name2)) +
    geom_net(aes(col = Gender), size = 6, linewidth = 1, fontsize = 3, labelcolour = "black", labelon = TRUE)
  
  
  
  # Adjusting the network ----
  # Tweak the network plot
  ggplot(data = mmnet, aes(from_id = Name1, to_id = Name2)) +
    geom_net(aes(col = Gender),
             size = 6,
             linewidth = 1,
             labelon = TRUE,
             fontsize = 3,
             labelcolour = "black",
             directed = TRUE) +
    scale_color_manual(values = c("#FF69B4", "#0099ff")) +
    ggmap::theme_nothing(legend = TRUE) +
    xlim(c(-0.05, 1.05)) +
    theme(legend.key = element_blank())
  
  
  
  # Autoplot on linear models ----
  # Create linear model: res
  res <- lm(Volume ~ Girth, data = trees)
  
  # Plot res
  plot(res)
  
  # Import ggfortify and use autoplot()
  library(ggfortify)
  autoplot(res, ncol = 2)
  
  
  # ggfortify - time series ----
  library(vars)
  data(Canada)
  # ggfortify and Canada are available
  
  # Inspect structure of Canada
  str(Canada)
  
  # Call plot() on Canada
  plot(Canada)
  
  # Call autoplot() on Canada
  autoplot(Canada)
  
  
  
  # !!! Distance matrices and Multi-Dimensional Scaling (MDS) ----
  # ggfortify and eurodist are available
  
  # Autoplot + ggplot2 tweaking
  autoplot(eurodist) + labs(x = "", y = "") +
    coord_fixed() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  
  # Autoplot of MDS
  autoplot(cmdscale(eurodist, eig = TRUE), label = TRUE, label.size = 3, size = 0)
  
  
  
  
  # !!! Plotting K-means clustering ----
  # Perform clustering
  iris_k <- kmeans(iris[-5], 3)
  
  # Autoplot: color according to cluster
  autoplot(iris_k, data = iris, frame = TRUE)
  
  # Autoplot: color according to species
  autoplot(iris_k, data = iris, frame = TRUE, col = 'Species')
  
  
  
  
  
  
  
# 3. Chapter 3: Plots for specific data types (Part 2)  ---- 
   # Working with maps from the maps package: USA ----
  # maps, ggplot2, and ggmap are pre-loaded
  
  # Use map_data() to create usa
  usa <- map_data("usa")
  data("us.cities")
  cities <- us.cities
  
  # Inspect structure of usa
  str(usa)
  
  # Build the map
  ggplot(usa, aes(x = long, y = lat, group = group)) +
    geom_polygon() +
    coord_map() +
    theme_nothing()
  
  
  # Working with maps from the maps package: adding points ----
  # usa, cities, and all required packages are available
  
  # Finish plot 1
  ggplot(usa, aes(x = long, y = lat, group = group)) +
    geom_polygon() +
    geom_point(data = cities, aes(group = country.etc, size = pop), col = "red", shape = 16,alpha = 0.6) +
    coord_map() +
    theme_map()
  
  # Arrange cities
  cities_arr <- arrange(cities, pop)
  
  # Copy-paste plot 1 and adapt
  ggplot(usa, aes(x = long, y = lat, group = group)) +
    geom_polygon(fill = "grey90") +
    geom_point(data = cities_arr, aes(group = country.etc, col = pop, size = 2), shape = 16,alpha = 0.6) +
    coord_map() +
    theme_map() +
    scale_color_viridis()
  
  
  
  # State choropleth ----
  # pop and all required packages are available
  
  # Use map_data() to create state
  state <- map_data("state")
  
  # Map of states
  ggplot(state, aes(x =long, y = lat, group = group, fill = region)) +
    geom_polygon(col = "white") +
    coord_map() +
    theme_nothing()
  
  # Merge state and pop: state2
  # läuft nicht, weil pop-data nicht verfügbar
  state2 <- merge(state, pop, by = "region")
  
  # Map of states with populations
  ggplot(state2, aes(x =long, y = lat, group = group, fill = Pop_est)) +
    geom_polygon(col = "white") +
    coord_map() +
    theme_map()
  
  
  # Map from shapefiles ----
  # http://www.gadm.org/country, download shapeshile ----
  # All required packages are available
  
  # Import shape information: germany
  germany <- rgdal::readOGR(dsn = "shapes_DEU", layer = "DEU_adm1")
  
  # fortify germany: bundes
  bundes <- fortify(germany)
  
  # Plot map of germany
  ggplot(bundes, aes(x = long, y = lat, group = group)) +
    geom_polygon(fill = "blue", col = "white") +
    coord_map() +
    theme_nothing()
  
  
  
  # Import shape information: georgia
  georgia <- rgdal::readOGR(dsn = "shapes_GEO", layer = "GEO_adm3")
  
  # fortify germany: bundes
  mizebi <- fortify(georgia)
  
  # Plot map of germany
  ggplot(mizebi, aes(x = long, y = lat, group = group)) +
    geom_polygon(fill = "blue", col = "white") +
    coord_map() +
    theme_nothing()
  
  
  
  # Choropleth from shapefiles ----
  # germany, bundes and unemp are available
  # läuft nicht, weil unemp fehlt
  
  # re-add state names to bundes
  bundes$state <- factor(as.numeric(bundes$id))
  levels(bundes$state) <- germany$NAME_1
  
  # Merge bundes and unemp: bundes_unemp
  bundes_unemp <- merge(bundes, unemp, by = "state")
  
  # Update the ggplot call
  ggplot(bundes_unemp, aes(x = long, y = lat, group = group, fill = unemployment)) +
    geom_polygon() +
    coord_map() +
    theme_map()
  
  
  # Different templates
  # Create london_map_13 with get_map
  london_map_13 <- get_map("London, England", zoom = 13)
  
  # Create the map of london
  ggmap(london_map_13)
  
  # Experiment with get_map() and use ggmap() to plot it!
  BadHomburg_map_13 <- get_map("Bad Homburg, Georgia", zoom = 13, maptype = "toner", source = "stamen")
  ggmap(BadHomburg_map_13)
  
  # Experiment with get_map() and use ggmap() to plot it!
  Tbilisi_map_13 <- get_map("Tbilisi, Georgia", zoom = 13, maptype = "hybrid")
  ggmap(Tbilisi_map_13)
  
  
  
  # Mapping points onto a cartographic map -----
    london_sites <- c(
    "Tower of London, London"             
    , "Buckingham Palace, London"           
    , "Tower Bridge, London"                
    , "Westminster Abby, London"            
    , "Queen Elizabeth Olympic Park, London"    
  )
  # london_sites and ggmap are available
  
  # Use geocode() to create xx
  xx <- geocode(london_sites)
  
  # Add a location column to xx
  xx$location <- sub(", London ", "", london_sites)
  
  # Get map data
  london_ton_13 <- get_map(location = "London, England", zoom = 13,
                           source = "stamen", maptype = "toner")
  
  # Add a geom_points layer
  ggmap(london_ton_13) +
    geom_point(data = xx, aes(col = location), size = 6)
  
  
  
  
  
  # Mapping points onto a cartographic map -----
  # london_sites and ggmap are available
  # Use geocode() to create xx
  xx <- geocode(london_sites)
  
  # Add a location column to xx
  xx$location <- sub(", London", "", london_sites)
  
  # Get map data
  london_ton_13 <- get_map(location = "London, England", zoom = 13,
                           source = "stamen", maptype = "toner")
  
  # Add a geom_points layer
  ggmap(london_ton_13) +
    geom_point(data = xx, aes(col = location), size = 6)
  
  
  
  
  
  
  
  # Using a bounding box ----
  # london_sites and ggmap are available
  
  # Build xx
  xx <- geocode(london_sites)
  xx$location <- sub(", London", "", london_sites)
  xx$location[5] <- "Queen Elizabeth\nOlympic Park"
  
  # Create bounding box: bbox
  bbox <- make_bbox(lon = xx$lon, lat = xx$lat, f = 0.3)
  
  # Update get_map to use bbox
  london_ton_13 <- get_map(location = bbox, zoom = 13,
                           source = "stamen", maptype = "toner")
  
  # Map from previous exercise
  ggmap(london_ton_13) +
    geom_point(data = xx, aes(col = location), size = 6)
  
  # New map with labels
  ggmap(london_ton_13) +
    geom_label(data = xx, aes(label = location), size = 4, fontface = "bold", fill = "grey90", col = "#E41A1C")
  
  
  
  
  
  # Combine cartographic and choropleth maps ----
  # bundes is available, as are all required packages (oben)
  
  # Get the map data of "Germany"
  germany_06 <- get_map("Germany", zoom = 6)
  
  # Plot map and polygon on top:
  ggmap(germany_06) +
    geom_polygon(data = bundes, aes(x = long, y = lat, group = group), fill = NA, col = "red") +
    coord_map()
  
  
  
  # The population pyramid ----
  # Läuft nicht: https://github.com/dgrtwo/gganimate/issues/22, vielleicht ältere Version von ImageMagick installieren
  # Inspect structure of japan
  str(japan)
  
  # Finish the code inside saveGIF
  saveGIF({
    
    # Loop through all time points
    for (i in unique(japan$time)) {
      
      # Subset japan: data
      data <- japan[japan$time == i,]
      
      # Finish the ggplot command
      p <- ggplot(data, aes(x = AGE, y = POP, fill = SEX, width = 1)) +
        coord_flip() +
        geom_bar(data = data[data$SEX == "Female",], stat = "identity") +
        geom_bar(data = data[data$SEX == "Male",], stat = "identity") +
        ggtitle(i)
      
      print(p)
      
    }
    
  }, movie.name = "pyramid.gif", interval = 0.1)
  
  
  
  
  # Animations with gganimate ----
  # Läuft nicht: https://github.com/dgrtwo/gganimate/issues/22, vielleicht ältere Version von ImageMagick installieren
  # Vocab, gganimate and ggplot2 are available

  # Update the static plot
  p <- ggplot(Vocab, aes(x = education, y = vocabulary, frame = year, cumulative = TRUE, color = year, group = year)) +
    stat_smooth(method = "lm", se = FALSE, size = 3)
  p
  
  # Call gg_animate on p
  gganimate(p, interval = 1.0, filename = "vocab.gif")
  
  

  
  
  
  
  
# 4. Chapter 4: ggplot2 Internals  ---- 
  # Viewport basics (1) ----
  # Draw rectangle in null viewport
  grid.rect(gp = gpar(fill = "grey90"))
  
  # Write text in null viewport
  grid.text("null viewport")
  
  # Draw a line
  grid.lines(x = c(0,0.75), y = c(0.25, 1), gp = gpar(lty = 2, col = "red"))
  
  
  # Viewport basics (2) ----
  # Create new viewport: vp
  vp <- viewport(x = 0.5, y = 0.5, width = 0.5, height = 0.5, just = "center")
  
  # Push vp
  pushViewport(vp)
  
  # Populate new viewport with rectangle
  grid.rect(gp = gpar(fill = "blue"))
  
  
  # !!!Build a plot from scratch (1) ----
  # Create plot viewport: pvp
  dev.off()
  mar <- c(5, 4, 2, 2)
  pvp <- plotViewport(mar)
  
  # Push pvp
  pushViewport(pvp)
  
  # Add rectangle
  grid.rect(gp = gpar(fill = "grey80"))
  
  # Create data viewport: dvp
  dvp <- dataViewport(xData = mtcars$wt, yData = mtcars$mpg)
  
  # Push dvp
  pushViewport(dvp)
  
  # Add two axes
  grid.xaxis()
  grid.yaxis()
  
  
  
  # Build a plot from scratch (2) ----
  # Add text to x axis
  grid.text("Weight", y = unit(-3, "lines"), name = "xaxis")
  
  # Add text to y axis
  grid.text("MPG", x = unit(-3, "lines"), rot = 90, name = "yaxis")
  
  # Add points
  grid.points(x = mtcars$wt, y = mtcars$mpg, pch = 16, name = "datapoints")
  
  
  
  # Modifying a plot with grid.edit ----
  # Edit "xaxis"
  grid.edit("xaxis", label = "Miles/(US) gallon")
  
  # Edit "yaxis"
  grid.edit("yaxis", label = "Weight (1000 lbs)")
  
  # Edit "datapoints"
  grid.edit("datapoints", gp = gpar(col = "#C3212766", cex = 2))
  
  
  
  # Exploring the gTable ----
  # A simple plot p
  p <- ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point()
  
  # Create gtab with ggplotGrob()
  gtab <- ggplotGrob(p)
  
  # Print out gtab
  gtab
  
  # Extract the grobs from gtab: gtab
  g <- gtab$grobs
  
  # Draw only the legend
  grid.draw(g[[15]])
  
  
  
  # Modifying the gTable ----
  # Code from before
  p <- ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point()
  gtab <- ggplotGrob(p)
  g <- gtab$grobs
  grid.draw(g[[15]])
  
  # Show layout of g[[8]]
  gtable_show_layout(g[[15]])
  
  # Create text grob
  my_text <- textGrob(label = "Motor Trend, 1974", gp = gpar(fontsize = 7, col = "gray25"))
  
  # Use gtable_add_grob to modify original gtab
  new_legend <- gtable_add_grob(gtab$grobs[[15]], my_text, 5, 3)
  
  # Update in gtab
  gtab$grobs[[15]] <- new_legend
  
  # Draw gtab
  grid.draw(gtab)
  
  
  
  # Exploring ggplot objects ----
  # Simple plot p
  p <- ggplot(mtcars, aes(x = wt, y = mpg, col = factor(cyl))) + geom_point()
  
  # Examine class() and names()
  class(p)
  names(p)
  
  # Print the scales sub-list
  p$scales$scales
  
  # Update p
  p <- p +
    scale_x_continuous("Length", limits = c(4, 8), expand = c(0, 0)) +
    scale_y_continuous("Width", limits = c(2, 4.5), expand = c(0, 0))
  
  # Print the scales sub-list
  p$scales$scales
  
  
  
  # ggplot_build and ggplot_table ----
  # Box plot of mtcars: p
  p <- ggplot(mtcars, aes(x = factor(cyl), y = wt)) + geom_boxplot()
  
  # Create pbuild
  pbuild <- ggplot_build(p)
  
  # a list of 3 elements
  names(pbuild)
  
  # Print out each element in pbuild
  pbuild$data
  pbuild$panel
  pbuild$plot
  
  # Create gtab from pbuild
  gtab <- ggplot_gtable(pbuild)
  
  # Draw gtab;
  # this should give the exact same output as pbuild$plot from before.
  grid.draw(gtab)
  
  
  
  # Extracting details ----
  # Box plot of mtcars: p
  p <- ggplot(mtcars, aes(x = factor(cyl), y = wt)) + geom_boxplot()
  
  # Build pdata
  pdata <- ggplot_build(p)$data
  
  # Access the first element of the list, a data frame
  pdata <- ggplot_build(p)$data
  class(pdata[[1]])
  
  # Isolate this data frame
  my_df <- pdata[[1]]
  
  # The x labels
  my_df$group <- ggplot_build(p)$panel$ranges[[1]]$x.labels
  
  # Print out specific variables
  my_df[c(1:6, 11)]
  
  
  
  # Arranging plots (1) ----
  # Add a theme (legend at the bottom)
  g1 <- ggplot(mtcars, aes(wt, mpg, col = cyl)) +
    geom_point(alpha = 0.5) +
    theme(legend.position = "bottom")
  
  # Add a theme (no legend)
  g2 <- ggplot(mtcars, aes(disp, fill = cyl)) +
    geom_histogram(position = "identity", alpha = 0.5, binwidth = 20) +
    theme(legend.position = "none")
  
  # Load gridExtra
  library(gridExtra)
  
  # Call grid.arrange()
  grid.arrange(g1,g2,ncol = 2)
  
  
  # Arranging plots (2) ----
  # Definitions of g1 and g2
  g1 <- ggplot(mtcars, aes(wt, mpg, col = cyl)) +
    geom_point() +
    theme(legend.position = "bottom")
  
  g2 <- ggplot(mtcars, aes(disp, fill = cyl)) +
    geom_histogram(binwidth = 20) +
    theme(legend.position = "none")
  
  # Extract the legend from g1
  my_legend <- ggplotGrob(g1)$grobs[[15]]  
  
  # Create g1_noleg
  g1_noleg <- g1 +
    theme(legend.position = "none")
  
  # Calculate the height: legend_height
  legend_height <- sum(my_legend$heights)
  
  # Arrange g1_noleg, g2 and my_legend
  grid.arrange(g1_noleg, g2, my_legend, layout_matrix = matrix(c(1, 3, 2, 3), ncol = 2), heights = unit.c(unit(1, "npc") - legend_height, legend_height))
  
  
  
  
  
  
# 5. Chapter 5: Data Munging and Visualization Case Study   ---- 
  # Case Study 1 - Bag Plot ---- 
  # Base package bag plot ----
  # läuft nicht, weil aplpack -Packet fehlt
  test_data <- data.frame(x = mtcars$wt, y = mtcars$mpg)
  
  # Call bagplot() on test_data
  # Create a bag plot of test_data data frame with the bagplot() function. The data frame only has two columns, so you won't need to specify the x and y arguments.
  bagplot(test_data)
  
  # Call compute.bagplot on test_data: bag
  # Call compute.bagplot() on test_data to obtain a named list of all the statistics used to build the bag plot and assign the results to an object called bag.
  bag <- compute.bagplot(test_data)
  
  # Display information
  # Code to print the hull loop and hull bag are included; add code to print out the outliers (pxy.outlier)
  bag$hull.loop
  bag$hull.bag
  bag$pxy.outlier
  
  # Highlight components
  # Code to add point layers to highlight two of the three aforementioned components in different colors is available. Can you add a points() function to show the outliers in a "purple" color?
  points(bag$hull.loop, col = "green", pch = 16)
  points(bag$hull.bag, col = "orange", pch = 16)
  points(bag$pxy.outlier, col = "purple", pch = 16)
  
  
  
  
  # Multilayer ggplot2 bag plot ----
  # bag and test_data are available
  
  # Create data frames from matrices
  hull.loop <- data.frame(x = bag$hull.loop[,1], y = bag$hull.loop[,2])
  hull.bag <- data.frame(x = bag$hull.bag[,1], y = bag$hull.bag[,2])
  pxy.outlier <- data.frame(x = bag$pxy.outlier[,1], y = bag$pxy.outlier[,2])
  
  # Finish the ggplot command
  ggplot(test_data, aes(x,  y)) +
    geom_polygon(data = hull.loop, fill = "green") +
    geom_polygon(data = hull.bag, fill = "orange") +
    geom_point(data = pxy.outlier, col = "purple", pch = 16, cex = 1.5)
  
  
  
  # Creating ggproto functions ----
  # ggproto for StatLoop (hull.loop)
  StatLoop <- ggproto("StatLoop", Stat,
                      required_aes = c("x", "y"),
                      compute_group = function(data, scales) {
                        bag <- compute.bagplot(x = data$x, y = data$y)
                        data.frame(x = bag$hull.loop[,1], y = bag$hull.loop[,2])
                      })
  
  # ggproto for StatBag (hull.bag)
  StatBag <- ggproto("StatBag", Stat,
                     required_aes = c("x", "y"),
                     compute_group = function(data, scales) {
                       bag <- compute.bagplot(x = data$x, y = data$y)
                       data.frame(x = bag$hull.bag[,1], y = bag$hull.bag[,2])
                     })
  
  # ggproto for StatOut (pxy.outlier)
  StatOut <- ggproto("StatOut", Stat,
                     required_aes = c("x", "y"),
                     compute_group = function(data, scales) {
                       bag <- compute.bagplot(x = data$x, y = data$y)
                       data.frame(x = bag$pxy.outlier[,1], y = bag$pxy.outlier[,2])
                     })
  
  
  
  # Creating stat_bag() ----
  # StatLoop, StatBag and StatOut are available
  
  # Combine ggproto objects in layers to build stat_bag()
  stat_bag <- function(mapping = NULL, data = NULL, geom = "polygon",
                       position = "identity", na.rm = FALSE, show.legend = NA,
                       inherit.aes = TRUE, loop = FALSE, ...) {
    list(
      # StatLoop layer
      layer(
        stat = StatLoop, data = data, mapping = mapping, geom = geom, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(na.rm = na.rm, alpha = 0.35, col = NA, ...)
      ),
      # StatBag layer
      layer(
        stat = StatBag, data = data, mapping = mapping, geom = geom, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(na.rm = na.rm, alpha = 0.35, col = NA,  ...)
      ),
      # StatOut layer
      layer(
        stat = StatOut, data = data, mapping = mapping, geom = "point", 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(na.rm = na.rm, alpha = 0.7, col = NA, shape = 21, ...)
      )
    )
  }
  
  
  
  # Use stat_bag() ----
  # hull.loop, hull.bag and pxy.outlier are available
  # stat_bag, test_data and test_data_2 are available
  
  # Previous method
  ggplot(test_data, aes(x = x,  y = y)) +
    geom_polygon(data = hull.loop, fill = "green") +
    geom_polygon(data = hull.bag, fill = "orange") +
    geom_point(data = pxy.outlier, col = "purple", pch = 16, cex = 1.5)
  
  # stat_bag
  ggplot(test_data, aes(x = x,  y = y)) +
    stat_bag(data = test_data, aes(x = x, y = y), fill = "black")
  
  # stat_bag on test_data_2
  ggplot(test_data2, aes(x = x,  y = y)) +
    stat_bag(data = test_data2, aes(x = x, y = y, fill = treatment))
  
  
  
  
  # Case Study II - Weather (Part 1) ----
  # Step 1: Read in data and examine ----
  # Import weather data
  weather <- read.fwf("NYNEWYOR.txt",
                      header = FALSE,
                      col.names = c("month", "day", "year", "temp"),
                      widths = c(14, 14, 13, 4))
  
  # Check structure of weather
  str(weather)
  
  # Create past with two filter() calls
  past <- weather %>%
    filter(!(month == 2 & day == 29)) %>%
    filter(year != max(year))
  
  # Check structure of past
  str(past)
  
  
  
  # Step 2: Summarize history ----
  # Create new version of past
  past <- past %>%
    group_by(year) %>%
    mutate(yearday = 1:length(day)) %>%
    ungroup %>%
    filter(!(temp == 99)) %>%
    group_by(yearday) %>%
    mutate(max = max(temp),
           min = min(temp),
           avg = mean(temp),
           CI_lower = Hmisc::smean.cl.normal(temp)[2],
           CI_upper = Hmisc::smean.cl.normal(temp)[3]) %>%
    ungroup()
  
  # Structure of past
  str(past)
  
  
  
  # Step 3: Plot history ----
  # Adapt historical plot
  ggplot(past, aes(x = yearday, y = temp)) +
    geom_point(col = "#EED8AE", alpha = 0.3, shape = 16) +
    geom_linerange(aes(ymin = CI_lower, ymax = CI_upper), col = "#8B7E66")
  
  
  
  # Step 4: Plot present ----
  # weather and past are available in your workspace
  
  # Create present
  present <- weather %>%
    filter(!(month == 2 & day == 29)) %>%
    filter(year == max(year)) %>%
    group_by(year) %>%
    mutate(yearday = 1:length(day)) %>%
    ungroup() %>%
    filter(temp != -99)
  
  # Add geom_line to ggplot command
  ggplot(past, aes(x = yearday, y = temp)) + 
    geom_point(col = "#EED8AE", alpha = 0.3, shape = 16) +
    geom_linerange(aes(ymin = CI_lower, ymax = CI_upper), col = "#8B7E66") + 
    geom_line(data = present)
  
  
  # Step 5: Find new record highs ----
  # Create past_highs
  past_highs <- past %>%
    group_by(yearday) %>%
    summarise(past_high = max(temp))
  
  # Create record_high
  record_high <- present %>%
    left_join(past_highs) %>%
    filter(temp > past_high)
  
  # Add record_high information to plot
  ggplot(past, aes(x = yearday, y = temp)) + 
    geom_point(col = "#EED8AE", alpha = 0.3, shape = 16) +
    geom_linerange(aes(ymin = CI_lower, ymax = CI_upper), col = "#8B7E66") +
    geom_line(data = present) +
    geom_point(data = record_high, col = "#CD2626")
  
  
  
  # Step 6: Efficiently calculate record highs and lows ----
  # Create past_extremes
  past_extremes <- past %>%
    group_by(yearday) %>%
    summarise(past_low = min(temp),
              past_high = max(temp))
  
  # Create record_high_low
  record_high_low <- present %>%
    left_join(past_extremes) %>%
    mutate(record = ifelse(temp < past_low, 
                           "#0000CD",
                           ifelse(temp > past_high, 
                                  "#CD2626", 
                                  "#00000000")))
  
  # Structure of record_high_low
  str(record_high_low)
  
  # Add point layer of record_high_low
  p <- ggplot(past, aes(x = yearday, y = temp)) + 
    geom_point(col = "#EED8AE", alpha = 0.3, shape = 16) +
    geom_linerange(aes(ymin = CI_lower, ymax = CI_upper), col = "#8B7E66") +
    geom_line(data = present) +
    geom_point(data = record_high_low, aes(col = record)) +
    scale_color_identity()
  p
  
  
  # Step 7: Custom legend ----
  # Finish the function draw_pop_legend
  draw_pop_legend <- function(x = 0.6, y = 0.2, width = 0.2, height = 0.2, fontsize = 10) {
    
    # Finish viewport() function
    pushViewport(viewport(x = x, y = y, width = width, height = height, just = "center"))
    
    legend_labels <- c("Past record high",
                       "95% CI range",
                       "Current year",
                       "Past years",
                       "Past record low")
    
    legend_position <- c(0.9, 0.7, 0.5, 0.2, 0.1)
    
    # Finish grid.text() function
    grid.text(label = legend_labels, x = 0.12, y = legend_position, 
              just = "left", gp = gpar(fontsize = fontsize, col = "grey20"))
    
    # Position dots, rectangle and line
    point_position_y <- c(0.1, 0.2, 0.9)
    point_position_x <- rep(0.06, length(point_position_y))
    grid.points(x = point_position_x, y = point_position_y, pch = 16,
                gp = gpar(col = c("#0000CD", "#EED8AE", "#CD2626")))
    grid.rect(x = 0.06, y = 0.5, width = 0.06, height = 0.4,
              gp = gpar(col = NA, fill = "#8B7E66"))
    grid.lines(x = c(0.03, 0.09), y = c(0.5, 0.5),
               gp = gpar(col = "black", lwd = 3))
    
    # Add popViewport() for bookkeeping
    popViewport()
  }
  
  # Print out plotting object p
  p
  
  # Call draw_pop_legend()
  draw_pop_legend()
  
  
  
  
  
  
  # !!!Case Study II - Weather (Part 2) ----
  # Step 1: clean_weather() ----
  # Finish the clean_weather function
  clean_weather <- function(file) {
    weather <- read.fwf(file,
                        header = FALSE,
                        col.names = c("month", "day", "year", "temp"),
                        widths = c(14, 14, 13, 4))
    weather %>%
      filter(!(month == 2 & day == 29)) %>%
      group_by(year) %>%
      mutate(yearday = 1:length(day)) %>%
      ungroup() %>%
      filter(temp != -99)
  }
  
  # Import NYNEWYOR.txt: my_data
  my_data <- clean_weather("NYNEWYOR.txt")
  
  
  # Step 2: Historical data ----
  # Create the stats object
  StatHistorical <- ggproto("StatHistorical", Stat,
                            compute_group = function(data, scales, params) {
                              data <- data %>%
                                filter(year != max(year)) %>%
                                group_by(x) %>%
                                mutate(ymin = Hmisc::smean.cl.normal(y)[3],
                                       ymax = Hmisc::smean.cl.normal(y)[2]) %>%
                                ungroup()
                            },
                            required_aes = c("x", "y", "year"))
  
  # Create the layer
  stat_historical <- function(mapping = NULL, data = NULL, geom = "point",
                              position = "identity", na.rm = FALSE, show.legend = NA, 
                              inherit.aes = TRUE, ...) {
    list(
      layer(
        stat = "identity", data = data, mapping = mapping, geom = geom,
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(na.rm = na.rm, col = "#EED8AE", alpha = 0.3, shape = 16, ...)
      ),
      layer(
        stat = StatHistorical, data = data, mapping = mapping, geom = "linerange",
        position = position, show.legend = show.legend, inherit.aes = inherit.aes,
        params = list(na.rm = na.rm, col = "#8B7E66", ...)
      )
    )
  }
  
  # Build the plot
  my_data <- clean_weather("NYNEWYOR.txt")
  ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
    stat_historical()
  
  
  # Step 3: Present data ----
  # Create the stats object
  StatPresent <- ggproto("StatPresent", Stat,
                         compute_group = function(data, scales, params) {
                           data <- filter(data, year == max(year))
                         },
                         required_aes =  c("x", "y", "year"))
  
  # Create the layer
  stat_present <- function(mapping = NULL, data = NULL, geom = "line",
                           position = "identity", na.rm = FALSE, show.legend = NA, 
                           inherit.aes = TRUE, ...) {
    layer(
      stat = StatPresent, data = data, mapping = mapping, geom = geom,
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, ...)
    )
  }
  
  # Build the plot
  my_data <- clean_weather("NYNEWYOR.txt")
  ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
    stat_historical() + 
    stat_present()
  
  
  # Step 4: Extremes ----
  # Create the stats object
  StatExtremes <- ggproto("StatExtremes", Stat,
                          compute_group = function(data, scales, params) {
                            
                            present <- data %>%
                              filter(year == max(year)) 
                            
                            past <- data %>%
                              filter(year != max(year)) 
                            
                            past_extremes <- past %>%
                              group_by(x) %>%
                              summarise(past_low = min(y),
                                        past_high = max(y))
                            
                            # transform data to contain extremes
                            data <- present %>%
                              left_join(past_extremes) %>%
                              mutate(record = ifelse(y < past_low, 
                                                     "#0000CD", 
                                                     ifelse(y > past_high, 
                                                            "#CD2626", 
                                                            "#00000000")))
                          },
                          required_aes = c("x", "y", "year"))
  
  # Create the layer
  stat_extremes <- function(mapping = NULL, data = NULL, geom = "point",
                            position = "identity", na.rm = FALSE, show.legend = NA, 
                            inherit.aes = TRUE, ...) {
    layer(
      stat = StatExtremes, data = data, mapping = mapping, geom = geom,
      position = position, show.legend = show.legend, inherit.aes = inherit.aes,
      params = list(na.rm = na.rm, ...)
    )
  }
  
  # Build the plot
  my_data <- clean_weather("NYNEWYOR.txt")
  ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
    stat_historical() +
    stat_present() +
    stat_extremes(aes(col = ..record..)) +
    scale_color_identity()
  
  
  # Step 5: Re-use plotting style ----
  # File paths of all datasets
  my_files <- c("NYNEWYOR.txt","FRPARIS.txt", "ILREYKJV.txt", "UKLONDON.txt")
  
  # Build my_data with a for loop
  my_data <- NULL
  for (file in my_files) {
    temp <- clean_weather(file)
    temp$id <- gsub(".txt","",file)
    my_data <- rbind(my_data, temp)
  }
  
  # Build the final plot, from scratch!
  ggplot(my_data, aes(x = yearday, y = temp, year = year)) +
    stat_historical() +
    stat_present() +
    stat_extremes(aes(col = ..record..)) +
    scale_color_identity() +
    facet_wrap(~id, ncol = 2)
  