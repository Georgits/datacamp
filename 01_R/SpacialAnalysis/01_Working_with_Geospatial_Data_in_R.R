# https://github.com/cwickham/geospatial

# 1. Chapter 1: Basic mapping with ggplot2 and ggmap  -----
    data("sales")
    data("ward_sales")
    data("preds")

# 1.1. Grabbing a background map ----
    library(ggmap)
    library(geospatial)
    corvallis <- c(lon = -123.2620, lat = 44.5646)
    
    # Get map at zoom level 5: map_5
    map_5 <- get_map(corvallis, zoom = 5, scale = 1)
    
    # Plot map at zoom level 5
    ggmap(map_5)
    
    # Get map at zoom level 13: corvallis_map
    corvallis_map <- get_map(corvallis, zoom = 13, scale = 1)
    
    # Plot map at zoom level 13
    ggmap(corvallis_map)
    
    # plot Bad Homburg map
    bad_homburg <- c(lon =  8.61885129999996, lat = 50.2259967)
    bad_homburg_map <- get_map(bad_homburg, zoom = 16, scale = 1)
    ggmap(bad_homburg_map)
    
    
    
# 1.2. Putting it all together -----
    # Look at head() of sales
    head(sales) # Diese Daten fehlen
    
    # Swap out call to ggplot() with call to ggmap()
    ggmap(corvallis_map) +
      geom_point(aes(lon, lat), data = sales)
    
# 1.3. Insight through aesthetics -----
    # Map color to year_built
    ggmap(corvallis_map) +
      geom_point(aes(lon, lat, color = year_built), data = sales)
    
    # Map size to bedrooms
    ggmap(corvallis_map) +
      geom_point(aes(lon, lat, size = bedrooms), data = sales)
    
    # Map color to price / finished_squarefeet
    ggmap(corvallis_map) +
      geom_point(aes(lon, lat, color = price / finished_squarefeet), data = sales)
    
    
# 1.4. Different maps -----
    corvallis <- c(lon = -123.2620, lat = 44.5646)
    
    # Add a maptype argument to get a satellite map
    corvallis_map_sat <- get_map(corvallis, zoom = 13, maptype = "satellite")
    
    # Edit to display satellite map
    ggmap(corvallis_map_sat) +
      geom_point(aes(lon, lat, color = year_built), data = sales)
    
    # Add source and maptype to get toner map from Stamen Maps
    corvallis_map_bw <- get_map(corvallis, zoom = 13, maptype = "toner", source = "stamen")
    
    # Edit to display toner map
    ggmap(corvallis_map_bw) +
      geom_point(aes(lon, lat, color = year_built), data = sales)
    
    
# 1.5. Leveraging ggplot2's strengths -----
    # Use base_layer argument to ggmap() to specify data and x, y mappings
    ggmap(corvallis_map_bw, base_layer =ggplot(sales, aes(lon, lat))) +
      geom_point(aes(color = year_built))
    
    # Use base_layer argument to ggmap() and add facet_wrap()
    ggmap(corvallis_map_bw, base_layer = ggplot(sales, aes(lon, lat))) +
      geom_point(aes(color = class)) +
      facet_wrap(~class)
    
# 1.6. A quick alternative -----
    qmplot(lon, lat, data = sales, 
           geom = "point", color = class) +
      facet_wrap(~ class)
    
    
# 1.7. Drawing polygons -----
    # Add a point layer with color mapped to ward
    ggplot(ward_sales, aes(lon, lat)) + 
      geom_point(aes(color = ward))
    
    # Add a point layer with color mapped to group
    ggplot(ward_sales, aes(lon, lat)) + 
      geom_point(aes(color = group))
    
    # Add a path layer with group mapped to group
    ggplot(ward_sales, aes(lon, lat)) + 
      geom_path(aes(group = group))
    
    # Add a polygon layer with fill mapped to ward, and group to group
    ggplot(ward_sales, aes(lon, lat)) + 
      geom_polygon(aes(fill = ward, group = group))
    

# 1.8. Choropleth map -----
    # Fix the polygon cropping
    ggmap(corvallis_map_bw, 
          base_layer = ggplot(ward_sales, aes(lon, lat)), extent = "normal", maprange = FALSE) +
      geom_polygon(aes(group = group, fill = ward))
    
    # Repeat, but map fill to num_sales
    ggmap(corvallis_map_bw, 
          base_layer = ggplot(ward_sales, aes(lon, lat)), extent = "normal", maprange = FALSE) +
      geom_polygon(aes(group = group, fill = num_sales))
    
    
    # Repeat again, but map fill to avg_price
    # Repeat, but map fill to num_sales
    ggmap(corvallis_map_bw, 
          base_layer = ggplot(ward_sales, aes(lon, lat)), extent = "normal", maprange = FALSE) +
      geom_polygon(aes(group = group, fill = avg_price), alpha = 0.8)

    
    
    
# 1.8. Raster data as a heatmap -----
    # Add a geom_point() layer
    ggplot(preds, aes(lon, lat))  + geom_point()
    
    # Add a tile layer with fill mapped to predicted_price
    ggplot(preds, aes(lon, lat)) +
      geom_tile(aes(fill = predicted_price))
    
    # Use ggmap() instead of ggplot()
    ggmap(corvallis_map_bw) +
      geom_tile(data = preds, aes(lon, lat, fill = predicted_price), alpha = 0.8)
  
    
# 2. Chapter 2: Point and polygon data  -----
    data("countries_sp")
    data("countries_spdf")
    data("tiny_countries_spdf")

# 2.1. Let's take a look at a spatial object ----
    library(sp)
    
    # Print countries_sp
    print(countries_sp)
    
    # Call summary() on countries_sp
    summary(countries_sp)
    
    # Call plot() on countries_sp
    plot(countries_sp)
    
# 2.2. What's inside a spatial object? ----
    # Call str() on countries_sp
    str(countries_sp)
    
    # Call str() on countries_sp with max.level = 2
    str(countries_sp, max.level = 2)
    
    
# 2.3. A more complicated spatial object ----
    # Call summary() on countries_spdf and countries_sp
    summary(countries_sp)
    summary(countries_spdf)
    
    # Call str() with max.level = 2 on countries_spdf
    str(countries_spdf, max.level = 2)
    
    # Plot countries_spdf
    plot(countries_spdf)
    
    
# 2.4. Walking the hierarchy ----
    # 169th element of countries_spdf@polygons: one
    one <- countries_spdf@polygons[[169]]
    
    # Print one
    print(one)
    
    # Call summary() on one
    summary(one)
    
    # Call str() on one with max.level = 2
    str(one, max.level =2)
    
    
# 2.5. Further down the rabbit hole ----
    one <- countries_spdf@polygons[[169]]
    
    # str() with max.level = 2, on the Polygons slot of one
    str(one@Polygons, max.level = 2)
    
    # str() with max.level = 2, on the 6th element of the one@Polygons
    str(one@Polygons[[6]], max.level = 2)
    
    
    # Call plot on the coords slot of 6th element of one@Polygons
    plot(one@Polygons[[6]]@coords)
    
    
# 2.6. Subsetting by index ----
    # Subset the 169th object of countries_spdf: usa
    usa <- countries_spdf[169,]
    
    # Look at summary() of usa
    summary(usa)
    
    # Look at str() of usa
    str(usa, max.level = 2)
    
    # Call plot() on usa
    plot(usa)
    
    
# 2.7. Accessing data in sp objects ----
    # Call head() and str() on the data slot of countries_spdf
    head(countries_spdf@data)
    str(countries_spdf@data)
    
    # Pull out the name column using $
    countries_spdf$name
    
    # Pull out the subregion column using [[
    countries_spdf[["subregion"]]
    
    
# 2.8. Subsetting based on data attributes ----
    # Create logical vector: is_nz
    is_nz <- countries_spdf$name =="New Zealand"
    
    # Subset countries_spdf using is_nz: nz
    nz <- countries_spdf[is_nz,]
    
    # Plot nz
    plot(nz)
    
    
# 2.9. tmap, a package that works with sp objects ----
    library(sp)
    library(tmap)
    
    # Use qtm() to create a choropleth map of gdp
    qtm(shp = countries_spdf, fill = "gdp") # läuft nicht, nicht klar, warum?
    
    
# 2.10. Building a plot in layers ----
    library(sp)
    library(tmap)
    data("Europe")
    
    # Add style argument to the tm_fill() call
    tm_shape(Europe) +
      tm_fill(col = "pop_est", style = "quantile") +
      # Add a tm_borders() layer 
      tm_borders(col = "burlywood4")
    
    # New plot, with tm_bubbles() instead of tm_fill()
    tm_shape(Europe) +
      tm_bubbles(size = "pop_est", style = "quantile") +
      # Add a tm_borders() layer 
      tm_borders(col = "burlywood4")
    

# 2.11. Why is Greenland so big? ----
    # library(sp)
    # library(tmap)
    data("World")
    
    # Switch to a Hobo–Dyer projection
    tm_shape(World, projection = "hd") +
      tm_grid(n.x = 11, n.y = 11) +
      tm_fill(col = "pop_est", style = "quantile")  +
      tm_borders(col = "burlywood4") 
    
    # Switch to a Robinson projection
    tm_shape(World, projection = "robin") +
      tm_grid(n.x = 11, n.y = 11) +
      tm_fill(col = "pop_est", style = "quantile")  +
      tm_borders(col = "burlywood4") 
    
    # Add tm_style_classic() to your plot
    tm_shape(World, projection = "robin") +
      tm_grid(n.x = 11, n.y = 11) +
      tm_fill(col = "pop_est", style = "quantile")  +
      tm_borders(col = "burlywood4") + 
      tm_style_classic()
    
 
# 2.12. Saving a tmap plot ----
    # library(sp)
    # library(tmap)
    
    # Plot from last exercise
    tm_shape(World, projection = "robin") +
      tm_grid(n.x = 11, n.y = 11) +
      tm_fill(col = "pop_est", style = "quantile")  +
      tm_borders(col = "burlywood4") + 
      tm_style_classic()
    
    # Save a static version "population.png"
    save_tmap(filename = "population.png")
    
    # Save an interactive version "population.html"
    save_tmap(filename = "population.html")
    
    
# 3. Chapter 3: Raster data and color  -----
    data("pop")
    data("pop_by_age")
    data("prop_by_age")
    data("migration")    
    data("land_cover")

    
# 3.1. What's a raster object? ----
    library(raster)
    data("pop")
    
    # Print pop
    print(pop)
    
    # Call str() on pop, with max.level = 2
    str(pop, max.level = 2)
    
    # Call summary on pop
    summary(pop)
    
# 3.2. Some useful methods ----
    # Call plot() on pop
    plot(pop)
    
    # Call str() on values(pop)
    str(values(pop))
    
    # Call head() on values(pop)
    head(values(pop))

    
# 3.3. A more complicated object ----
    # Print pop_by_age
    print(pop_by_age)
    
    # Subset out the under_1 layer using [[
    pop_by_age[["under_1"]]
    
    # Plot the under_1 layer
    plot(pop_by_age[["under_1"]])
    
    
# 3.4. A package that uses Raster objects ----
    library(tmap)
    
    # Specify pop as the shp and add a tm_raster() layer
    tm_shape(pop) +
      tm_raster()
    
    # Plot the under_1 layer in pop_by_age
    tm_shape(pop_by_age) +
      tm_raster(col  = "under_1")

    library(rasterVis)
    # Call levelplot() on pop
    levelplot(pop)
    
    
# 3.5. Adding a custom continuous color palette to ggplot2 plots ----
    library(RColorBrewer)
    # 9 steps on the RColorBrewer "BuPu" palette: blups
    blups <- brewer.pal(9, "BuPu")
    
    # Add scale_fill_gradientn() with the blups palette
    ggplot(preds) +
      geom_tile(aes(lon, lat, fill = predicted_price), alpha = 0.8) +
      scale_fill_gradientn(colors = blups)

    library(viridisLite)
    # viridisLite viridis palette with 9 steps: vir 
    vir <- viridis(n = 9)

    # Add scale_fill_gradientn() with the vir palette
    ggplot(preds) +
      geom_tile(aes(lon, lat, fill = predicted_price), alpha = 0.8) +
      scale_fill_gradientn(colors = vir)
    
    # mag: a viridisLite magma palette with 9 steps
    mag <- magma(n = 9)
    
    # Add scale_fill_gradientn() with the mag palette
    ggplot(preds) +
      geom_tile(aes(lon, lat, fill = predicted_price), alpha = 0.8)  +
      scale_fill_gradientn(colors = mag)
    

# 3.6. Custom palette in tmap ----
    # Generate palettes from last time
    library(RColorBrewer)
    blups <- brewer.pal(9, "BuPu")
    
    library(viridisLite)
    vir <- viridis(9)
    mag <- magma(9)
    
    # Use the blups palette
    tm_shape(prop_by_age) +
      tm_raster(col = "age_18_24", palette = blups) +
      tm_legend(position = c("right", "bottom"))
    
    # Use the vir palette
    tm_shape(prop_by_age) +
      tm_raster(col = "age_18_24", palette = vir) +
      tm_legend(position = c("right", "bottom"))
    
    # Use the mag palette but reverse the order
    tm_shape(prop_by_age) +
      tm_raster(col = "age_18_24", palette = rev(mag)) +
      tm_legend(position = c("right", "bottom"))
    
    
# 3.7. An interval scale example ----
    library(tmap)
    library(viridisLite)
    mag <- viridisLite::magma(7)
    
    library(classInt)
    
    # Create 5 "pretty" breaks with classIntervals()
    classIntervals(values(prop_by_age[["age_18_24"]]), n = 5, style = "pretty")
    
    # Create 5 "quantile" breaks with classIntervals()
    classIntervals(values(prop_by_age[["age_18_24"]]), n = 5, style = "quantile")
    
    # Use 5 "quantile" breaks in tm_raster()
    tm_shape(prop_by_age) +
      tm_raster("age_18_24", palette = mag, n = 5, style = "quantile") +
      tm_legend(position = c("right", "bottom"))
    
    # Create histogram of proportions
    hist(values(prop_by_age[["age_18_24"]]))
    
    # Use fixed breaks in tm_raster()
    tm_shape(prop_by_age) +
      tm_raster("age_18_24", palette = mag,
                style = "fixed", breaks = c(0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 1))
    
    # Save your plot to "prop_18-24.html"
    save_tmap(filename = "prop_18-24.html")
    
    
# 3.8. A diverging scale example ----
    # Print migration
    print(migration)
    
    # Diverging "RdGy" palette
    red_gray <- brewer.pal(7, "RdGy")
    
    # Use red_gray as the palette 
    tm_shape(migration) +
      tm_raster(palette = red_gray) +
      tm_legend(outside = TRUE, outside.position = c("bottom"))
    
    # Add fixed breaks 
    tm_shape(migration) +
      tm_raster(palette = red_gray, style = "fixed", breaks = c(-5e6, -5e3, -5e2, -5e1, 5e1, 5e2, 5e3, 5e6)) +
      tm_legend(outside = TRUE, outside.position = c("bottom"))
    
    
# 3.9. A qualitative example ----
    library(raster)
    
    # Plot land_cover
    plot(land_cover)
    tm_shape(land_cover) +
      tm_raster()
    
    # Palette like the ggplot2 default
    hcl_cols <- hcl(h = seq(15, 375, length = 9), 
                    c = 100, l = 65)[-9]
    
    # Use hcl_cols as the palette
    tm_shape(land_cover) +
      tm_raster(palette = hcl_cols)
    
    
    # Examine levels of land_cover
    levels(land_cover)
    
    # A set of intuitive colors
    intuitive_cols <- c(
      "darkgreen",
      "darkolivegreen4",
      "goldenrod2",
      "seagreen",
      "wheat",
      "slategrey",
      "white",
      "lightskyblue1"
    )
    
    # Use intuitive_cols as palette
    tm_shape(land_cover) +
      tm_raster(palette = intuitive_cols) +
      tm_legend(position = c("left", "bottom"))
    
    
# 4. Chapter 4: Data import and projections  -----
    data("neighborhoods")
    data("nyc_income")
    data("water")
    data("income_grid")
    
# 4.1. Reading in a shapefile ----
    library(sp)
    library(rgdal)
    
    # Use dir() to find directory name
    # download from: https://www1.nyc.gov/site/planning/data-maps/open-data/dwn-nynta.page
    # Source: Open Data Platform of the Department of City Planning
    dir()
    
    # Call dir() with directory name
    dir("nynta_17a")
    
    # Read in shapefile with readOGR(): neighborhoods
    neighborhoods <- readOGR("nynta_17a", "nynta")
    
    # summary() of neighborhoods
    summary(neighborhoods)
    
    # Plot neighboorhoods
    plot(neighborhoods)
    
# 4.2. Reading in a raster file ----
    library(raster) 
    
    # Call dir()
    dir()
    
    # Call dir() on the directory
    dir("nyc_grid_data")
    
    # Use raster() with file path: income_grid
    income_grid <- raster("nyc_grid_data/m5602ahhi00.tif")
    
    # Call summary() on income_grid
    summary(income_grid)
    
    # Call plot() on income_grid
    plot(income_grid)
    
    
# 4.3. Getting data using a package ----
    library(sp)
    library(tigris)
    
    # Call tracts(): nyc_tracts
    nyc_tracts <- tracts(state = "NY", county = "New York", cb = TRUE)
    
    # Call summary() on nyc_tracts
    summary(nyc_tracts)
    
    # Plot nyc_tracts
    plot(nyc_tracts)
    
# 4.4. Merging data from different CRS/projections ----
    library(sp)
    
    # proj4string() on nyc_tracts and neighborhoods
    proj4string(nyc_tracts)
    proj4string(neighborhoods)
    
    # coordinates() on nyc_tracts and neighborhoods
    head(coordinates(nyc_tracts))
    head(coordinates(neighborhoods))
    
    # plot() neighborhoods and nyc_tracts
    plot(neighborhoods)
    plot(nyc_tracts, col = "red", add  = TRUE)
    # Why didn't we see the tracts on our plot of neighborhoods? Simply because the coordinates of the tracts put them way off the boundaries of our plot. 
    
# 4.5. Merging data from different CRS/projections ----
    library(sp)
    library(raster)
    
    # Use spTransform on neighborhoods: neighborhoods
    neighborhoods <- spTransform(neighborhoods, proj4string(nyc_tracts))
    
    # head() on coordinates() of neighborhoods
    head(coordinates(neighborhoods))
    
    # Plot neighborhoods, nyc_tracts and water
    plot(neighborhoods)
    plot(nyc_tracts, col = "red", add  = TRUE)
    plot(water, col = "blue", add  = TRUE)
    # If you plot the untransformed objects with tmap, it actually does this transformation on the fly, but it's useful to know how to do it manually. 
    
# 4.6. The wrong way ----
    # library(sp)
    # 
    # # Use str() on nyc_income and nyc_tracts@data
    # str(nyc_income)
    # str(nyc_tracts@data)
    # 
    # # Highlight tract 002201 in nyc_tracts
    # plot(nyc_tracts)
    # plot(nyc_tracts[nyc_tracts$TRACTCE == "002201", ], 
    #      col = "red", add = TRUE)
    # 
    # # Set nyc_tracts@data to nyc_income
    # nyc_tracts@data <- nyc_income
    # 
    # # Highlight tract 002201 in nyc_tracts
    # plot(nyc_tracts)
    # plot(nyc_tracts[nyc_tracts$tract == "002201", ], 
    #      col = "red", add = TRUE)
  
# 4.7. Checking data will match ----
    # Check for duplicates in nyc_income
    any(duplicated(nyc_income$tract))
    
    # Check for duplicates in nyc_tracts
    any(duplicated(nyc_tracts$TRACTCE))
    
    # Check nyc_tracts in nyc_income
    all(nyc_tracts$TRACTCE %in% nyc_income$tract)
    
    # Check nyc_income in nyc_tracts
    all(nyc_income$tract %in% nyc_tracts$TRACTCE)
    
    
# 4.8. Merging data attributes ----
    library(sp)
    library(tmap)
    
    # Merge nyc_tracts and nyc_income: nyc_tracts_merge
    nyc_tracts_merge <- merge(nyc_tracts, nyc_income, by.x = "TRACTCE", by.y = "tract")
    
    # Call summary() on nyc_tracts_merge
    summary(nyc_tracts_merge)
    
    # Choropleth with col mapped to estimate
    tm_shape(nyc_tracts_merge) + 
      tm_fill(col  = "estimate")
    
    
# 4.9. A first plot ----
    library(tmap)
    
    tm_shape(nyc_tracts_merge) +
      tm_fill(col = "estimate") +
      # Add a water layer, tm_fill() with col = "grey90"
      tm_shape(water)  +
      tm_fill(col = "grey90") +
      # Add a neighborhood layer, tm_borders()
      tm_shape(neighborhoods) +
      tm_borders()
    
# 4.10. Subsetting the neighborhoods ----
    library(tmap)
    
    # Find unique() nyc_tracts_merge$COUNTYFP
    unique(nyc_tracts_merge$COUNTYFP)
    
    # Add logical expression to pull out New York County
    manhat_hoods <- neighborhoods[neighborhoods$CountyFIPS == "061", ]
    
    tm_shape(nyc_tracts_merge) +
      tm_fill(col = "estimate") +
      tm_shape(water) +
      tm_fill(col = "grey90") +
      # Edit to use manhat_hoods instead
      tm_shape(manhat_hoods) +
      tm_borders() +
      # Add a tm_text() layer
      tm_text(text = "NTAName")  
    
    
# 4.11. Adding neighborhood labels ----
    library(tmap)
    
    # gsub() to replace " " with "\n"
    manhat_hoods$name <- gsub(" ", "\n", manhat_hoods$NTAName)
    
    # gsub() to replace "-" with "/\n"
    manhat_hoods$name <- gsub("-", "/\n", manhat_hoods$name)
    
    # Edit to map text to name, set size to 0.5
    tm_shape(nyc_tracts_merge) +
      tm_fill(col = "estimate") +
      tm_shape(water) +
      tm_fill(col = "grey90") +
      tm_shape(manhat_hoods) +
      tm_borders() +
      tm_text(text = "name", size = 0.5)
    
    
# 4.12. Tidying up the legend and some final tweaks ----
    library(tmap)
    
    tm_shape(nyc_tracts_merge) +
      # Add title and change palette
      tm_fill(col = "estimate", 
              title = "Median Income",
              palette = "Greens") +
      # Add tm_borders()
      tm_borders(col = "grey60", lwd = 0.5) +
      tm_shape(water) +
      tm_fill(col = "grey90") +
      tm_shape(manhat_hoods) +
      # Change col and lwd of neighborhood boundaries
      tm_borders(col = "grey40", lwd = 2) +
      tm_text(text = "name", size = 0.5) +
      # Add tm_credits()
      tm_credits("Source: ACS 2014 5-year Estimates, \n accessed via acs package",position = c("right", "bottom"))
    
    
    # Save map as "nyc_income_map.png"
    save_tmap(filename = "nyc_income_map.png", width = 4, height = 7)
    save_tmap(filename = "nyc_income_map.html", width = 4, height = 7)
    