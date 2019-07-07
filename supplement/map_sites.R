
library(httr)
library(ggplot2)
library(dplyr)
library(sf)
library(USAboundaries)
library(maps)

lake_geo <- readRDS('~/Downloads/lakes_sp.rds')
lake_metadata <- read.csv('supplement/in/lake_metadata_table.csv')

lake_geo <- st_as_sf(lake_geo)


getLakes <- function(){
  # shapefile_loc <- "http://geo.glin.net/gis/shps/glin_gl_mainlakes.zip"
  #
  # destination = file.path(tempdir(),"glin_gl_mainlakes.zip")
  # file <- GET(shapefile_loc, write_disk(destination, overwrite=T))
  # filePath <- tempdir()
  # unzip(destination, exdir = filePath)

  lakes <- st_read('~/Downloads/main_lakes_SHP', layer = "main_lakes")
  return(lakes)
}

grlakes <- getLakes()

mapRange <- c(-97.6, -86.7,43, 49)
#streamorder <- 5
crsLONGLAT <- 4326
crs_plot <- st_crs(102003)


# set the state and county names of interest
state_names <- c("minnesota","north dakota", "south dakota", "iowa",
                 "michigan","illinois", 'wisconsin', 'nebraska')

# get STATE data
my_states <- us_states(resolution = "high", states = state_names) %>%
  st_transform(crs = crsLONGLAT)

canada <- ggplot2::map_data('world', 'canada' ) %>%
  st_as_sf(coords = c("long","lat"),
           crs = crsLONGLAT)

lakes_test <- ggplot2::map_data('lakes') %>%
  st_as_sf(coords = c("long","lat"),
           crs = crsLONGLAT)



bb <- st_sfc(
  st_point(mapRange[c(1,3)]),
  st_point(mapRange[c(2,4)]),
  crs = crsLONGLAT)

bb_proj <- st_transform(bb, crs = crs_plot)
b <- st_bbox(bb_proj)

# read in sites we want to plot
sites_df <- st_as_sf(lake_metadata[, c('nhd_id', 'lake_name', 'latitude', 'longitude')],
                     coords = c("longitude","latitude"),
                     crs = crsLONGLAT) %>%
  st_transform(crs = crs_plot)

# create data frame of sparkling and mendota
target_lakes <- filter(sites_df, lake_name %in% c('Lake Mendota', 'Sparkling Lake'))
lake_labels <-data.frame(lake_name = target_lakes$lake_name,
                         x = st_coordinates(target_lakes)[,1],
                         y = st_coordinates(target_lakes)[,2])

sites_proj <- st_transform(sites_df, crs = crs_plot)
target_lakes_proj <- st_transform(target_lakes, crs = crs_plot)


# Make a quick Map:

#base_map <-
data("wrld_simpl", package = "maptools")
library(ggrepel)
wrld_simple <- st_as_sf(wrld_simpl) %>%
  st_transform(crs = crs_plot)
first_pass <- ggplot() +
  geom_sf(data = wrld_simple, color = 'gray90', fill = 'gray90') +
  geom_sf(data = my_states, fill='gray90', color = 'white', size = 1.2) +
  geom_sf(data = grlakes, fill = 'gray80', color = 'gray80') +
  geom_sf(data = lake_geo, fill = "gray70", color = NA) +
  #geom_sf(data = lakes_test, fill = "gray70", color = NA) +
  geom_sf(data = sites_proj, shape = 21, alpha = 0.7, fill = 'black', color = 'white') +
  geom_sf(data = target_lakes, shape = 21, fill = 'red', color = 'white') +
  geom_text_repel(data = lake_labels, aes(label = lake_name, x = x, y = y), point.padding = 0.5, color = 'red') +
  coord_sf(crs = crs_plot,
           xlim = c(b["xmin"],b["xmax"]),
           ylim = c(b["ymin"],b["ymax"])) +
  theme(panel.grid = element_blank()) +
  theme_minimal() +
  labs(x = '', y = '')

ggsave('figures/WRR_S1_map.png', first_pass, height = 6, width = 6)

