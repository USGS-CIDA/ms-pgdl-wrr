
get_rmse_data <- function(metadata_file) {
  lake_names <- read.csv(metadata_file) %>%
    select(site_id = nhd_id, lake_name)

  temp_file_loc <- tempfile(fileext = '.csv')
  item_file_download('5d925048e4b0c4f70d0d0596', names = 'all_sparse_RMSE.csv',
                     destinations = temp_file_loc)

  temp_file_loc2 <- tempfile(fileext = '.csv')
  item_file_download('5d925048e4b0c4f70d0d0596', names = 'all_RMSE.csv',
                     destinations = temp_file_loc2)

  sparse_rmse_dat <- readr::read_csv(temp_file_loc)
  full_rmse_dat <- readr::read_csv(temp_file_loc2)

  rmse_dat <- bind_rows(sparse_rmse_dat, full_rmse_dat) %>%
    group_by(site_id, exper_id, model_type) %>%
    summarize(mean_rmse = mean(rmse))

  rmse_dat <- left_join(rmse_dat, lake_names)

  # rmse_order <- filter(rmse_dat, exper_id == 'historical_all') %>%
  #   filter(model_type %in% c('pb', 'pgdl')) %>%
  #   tidyr::spread(model_type, mean_rmse) %>%
  #   mutate(rmse_diff = pb - pgdl) %>%
  #   arrange(rmse_diff)


  rmse_order <- filter(rmse_dat, exper_id == 'historical_all') %>%
    filter(model_type %in% c('pgdl')) %>%
    arrange(mean_rmse)


  rmse_dat$lake_name <- factor(rmse_dat$lake_name, levels = rmse_order$lake_name)

  return(rmse_dat)

}

plot_rmse_dat <- function(file_out, metadata_file) {

  rmse_dat <- get_rmse_data(metadata_file = metadata_file)


  mod_10 <- filter(rmse_dat, exper_id %in% c('historical_010', 'historical_10'))
  mod_10$model <- factor(mod_10$model_type, levels = c('pb', "dl", "pgdl"))
  mod_other <- filter(rmse_dat, exper_id %in% c('historical_all'))
  mod_other$model <- factor(mod_other$model_type, levels = c("pb0", "pb","dl","pgdl"))

  p1 <- ggplot(mod_other, aes(x = lake_name, y = mean_rmse)) +
    geom_point(aes(color = model, shape = model, fill = model), size = 1.7, alpha = 0.8) +
    scale_shape_manual(values = c(1,21,22,23),
                       labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    scale_color_manual(values = c('#1b9e77','#1b9e77','#d95f02', '#7570b3'),
                       labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    scale_fill_manual(values = c('#1b9e77','#1b9e77','#d95f02', '#7570b3'),
                      labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    #scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
    #facet_grid(rows = vars(model_10)) +
    theme_bw() +
    coord_cartesian(ylim = c(0.9, 5.2)) +
    theme(
      #axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = c(0.055,0.17),
      panel.grid.minor.y = element_blank(),
      panel.grid.minor.x = element_blank(),
      legend.title = element_blank(),
      legend.background = element_rect(colour = 'black',
                                       fill = 'white', linetype='solid'),
      legend.text = element_text(size = 8),
      legend.key.height = unit(0.5, 'line'),
      axis.text.x = element_blank(),
      plot.margin = unit(c(0,0,0,0),'cm')) +
    labs(x = '', y = 'RMSE') +
    scale_y_reverse()

  p2 <- ggplot(mod_10, aes(x = lake_name, y = mean_rmse)) +
    geom_point(aes(color = model, shape = model, fill = model), size = 1.7, alpha = 0.8) +
    scale_shape_manual(values = c(21,22,23),
                       labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    scale_color_manual(values = c('#1b9e77','#d95f02', '#7570b3'),
                       labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    scale_fill_manual(values = c('#1b9e77','#d95f02', '#7570b3'),
                      labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    #scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
    #facet_grid(rows = vars(model_10)) +
    theme_bw() +
    coord_cartesian(ylim = c(0.9, 5.2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          legend.position = c(0.055,0.16),
          panel.grid.minor.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          legend.title = element_blank(),
          legend.background = element_rect(colour = 'black',
                                           fill = 'white', linetype='solid'),
          legend.text = element_text(size = 8),
          legend.key.height = unit(0.5, 'line'),
          plot.margin = unit(c(0,0,0,0),'cm')) +
    labs(x = '', y = 'RMSE') +
    scale_y_reverse()

  p <- cowplot::plot_grid(p1, p2, ncol = 1, rel_heights = c(0.4, 0.56))

  ggsave(file_out, p, height = 7, width = 10, units = 'in')
}


