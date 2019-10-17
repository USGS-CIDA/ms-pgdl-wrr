



make_map <- function(file_out, metadata_file, shp_file) {

  # bring in great lakes for map
  # originally from "http://geo.glin.net/gis/shps/glin_gl_mainlakes.zip"
  grlakes <- st_read(shp_file)

  # study lake metdata
  lake_metadata <- read.csv(metadata_file)


  # set map features
  mapRange <- c(-97.6, -86.7,43, 49)
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


  # Make a map

  data("wrld_simpl", package = "maptools")
  wrld_simple <- st_as_sf(wrld_simpl) %>%
    st_transform(crs = crs_plot)

  # this creates the map from the supplement without the lake shapes
  p <- ggplot() +
    geom_sf(data = wrld_simple, color = 'gray90', fill = 'gray90') +
    geom_sf(data = my_states, fill='gray90', color = 'white', size = 1.2) +
    geom_sf(data = grlakes, fill = 'gray80', color = 'gray80') +
    geom_sf(data = sites_proj, shape = 21, alpha = 0.7, fill = 'black', color = 'white') +
    geom_sf(data = target_lakes, shape = 21, fill = 'red', color = 'white') +
    geom_text_repel(data = lake_labels, aes(label = lake_name, x = x, y = y), point.padding = 0.5, color = 'red') +
    coord_sf(crs = crs_plot,
             xlim = c(b["xmin"],b["xmax"]),
             ylim = c(b["ymin"],b["ymax"])) +
    theme(panel.grid = element_blank()) +
    theme_minimal() +
    labs(x = '', y = '')

  ggsave(file_out, p, height = 6, width = 6)

}








