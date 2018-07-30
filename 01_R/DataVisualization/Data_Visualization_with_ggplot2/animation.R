# http://blog.revolutionanalytics.com/2016/02/japans-ageing-population-animated-with-r.html
# https://github.com/walkerke/idbr
# Läuft nicht: https://github.com/dgrtwo/gganimate/issues/22

library(idbr) # devtools::install_github('walkerke/idbr')
library(ggplot2)
library(animation)
library(dplyr)
library(ggthemes)
library(ImageMagick)

idb_api_key("1cfba561ff6d00351e4e7ea2264eb8b8760b6c62")

male <- idb1('JA', 2010:2050, sex = 'male') %>%
  mutate(POP = POP * -1,
         SEX = 'Male')

female <- idb1('JA', 2010:2050, sex = 'female') %>%
  mutate(SEX = 'Female')

japan <- rbind(male, female) %>%
  mutate(abs_pop = abs(POP))

# Animate it with a for loop

saveGIF({
  
  for (i in 2010:2050) {
    
    title <- as.character(i)
    
    year_data <- filter(japan, time == i)
    
    g1 <- ggplot(year_data, aes(x = AGE, y = POP, fill = SEX, width = 1)) +
      coord_fixed() + 
      coord_flip() +
      annotate('text', x = 98, y = -800000, 
               label = 'Data: US Census Bureau IDB; idbr R package', size = 3) + 
      geom_bar(data = subset(year_data, SEX == "Female"), stat = "identity") +
      geom_bar(data = subset(year_data, SEX == "Male"), stat = "identity") +
      scale_y_continuous(breaks = seq(-1000000, 1000000, 500000),
                         labels = paste0(as.character(c(seq(1, 0, -0.5), c(0.5, 1))), "m"), 
                         limits = c(min(japan$POP), max(japan$POP))) +
      theme_economist(base_size = 14) + 
      scale_fill_manual(values = c('#ff9896', '#d62728')) + 
      ggtitle(paste0('Population structure of Japan, ', title)) + 
      ylab('Population') + 
      xlab('Age') + 
      theme(legend.position = "bottom", legend.title = element_blank()) + 
      guides(fill = guide_legend(reverse = TRUE))
    
    print(g1)
    
  }
  
}, movie.name = 'japan_pyramid.gif', interval = 0.1, ani.width = 700, ani.height = 600)