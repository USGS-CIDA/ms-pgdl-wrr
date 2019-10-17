# function to read a test file and identify the corresponding predictions
get_test_dat <- function() {
  me_file_loc <- tempfile(fileext = '.csv')
  item_file_download('5d925066e4b0c4f70d0d0599', names = 'me_test.csv',
                       destinations = me_file_loc)

  me_dat <- readr::read_csv(me_file_loc) %>%
    mutate(lake = 'me')

  sp_file_loc <- tempfile(fileext = '.csv')
  item_file_download('5d92507be4b0c4f70d0d059b', names = 'sp_test.csv',
                     destinations = sp_file_loc)

  sp_dat <- readr::read_csv(sp_file_loc) %>%
    mutate(lake = 'sp')

  return(bind_rows(me_dat, sp_dat))
}

prep_pred_obs <- function(test_obs, model_preds) {

  # match up preds to test_obs, interpolating GLM predictions to match the observation depths
  pred_obs <- bind_rows(lapply(unique(test_obs$date), function(dt) {
    pred_1d <- filter(model_preds, date == dt, !is.na(depth))

    obs_1d <- filter(test_obs, date == dt) %>%
      rename(obs = temp)
    tryCatch({
      if(nrow(pred_1d) == 0) stop(sprintf('no predictions on %s', dt))
      if(min(pred_1d$Depth) != 0) warning(sprintf('no GLM prediction at 0m on %s', dt))
      mutate(obs_1d, pred = approx(x=pred_1d$depth, y=pred_1d$pred, xout=obs_1d$depth, rule=1)$y)
    }, error=function(e) {
      message(sprintf('approx failed for mendota on %s: %s', dt, e$message))
      mutate(obs_1d, pred = NA)
    })
  }))


  return(pred_obs)
}
get_bias_dat <- function(test_data, lake, experiment, type){
  item_locations <- data.frame(lake = c('me', 'sp'),
                               items = c('5d915cb2e4b0c4f70d0ce523', '5d915cc6e4b0c4f70d0ce525'),
                               stringsAsFactors = FALSE)
  file_prefix <- paste0(lake, '_', type, '_predict_')
  models <- c('pgdl', 'dl', 'pb')

  pred_obs <- list()

  for (i in 1:length(models)) {
    temp_file <- paste0(file_prefix, models[i], '.csv')
    temp_file_loc <- tempfile(fileext = '.csv')
    temp_item <- item_locations$items[item_locations$lake == lake]

    item_file_download(temp_item, names = temp_file,
                       destinations = temp_file_loc)

    temp_pred_data <- readr::read_csv(temp_file_loc) %>%
      gather(depth_code, temp, -date, -exper_n, -exper_id) %>%
      mutate(depth = as.numeric(substring(depth_code, 6)), lake = lake, model_type = models[i]) %>%
      select(date, depth, pred = temp, exper_n, exper_id, lake, model_type) %>%
      arrange(date) %>%
      filter(exper_n == experiment)


    filt_test_dat <- test_dat <- test_data[test_data$lake %in% lake,] %>%
      filter(exper_type == type) %>%
      filter(exper_n == experiment) %>%
      filter(!is.na(temp))

    pred_obs[[i]] <- prep_pred_obs(filt_test_dat, temp_pred_data) %>%
      mutate(model_type = models[i])

  }

  all_pred_obs <- bind_rows(pred_obs)

  depth_max <- ifelse(lake == 'sp', 18, 20)

  dat_to_add <- all_pred_obs %>%
    select(date, depth, obs) %>%
    distinct() %>%
    rename(bias = obs) %>%
    mutate(model_type = 'Observed')

  bias_dat <- all_pred_obs %>%
    mutate(bias = pred - obs) %>%
    bind_rows(dat_to_add) %>%
    mutate(depth_bin = cut(depth, breaks = seq(0, depth_max, by = 2),
                           include.lowest = TRUE)) %>%
    filter(!is.na(depth_bin)) %>%  # exclude Mendota depths that are >depth_max (20m)
    group_by(date, depth_bin, model_type) %>%
    summarize(median_bias = median(bias)) %>%
    mutate(doy = lubridate::yday(date), year = lubridate::year(date))

  # change levels
  bias_dat$model_type <- factor(bias_dat$model_type, levels = c('Observed', 'pb', 'dl', 'pgdl'))
  levels(bias_dat$model_type) <- c('Observed', 'PB bias', 'DL bias', 'PGDL bias')

  # find min and max for bias scales
  bias_scales <- filter(bias_dat, model_type != 'Observed')
  bias_range <- range(bias_scales$median_bias, na.rm = TRUE)
  bias_low <- floor(bias_range[1]) - 1
  bias_high <- ceiling(bias_range[2]) + 1

  # add dummy obs to get scales the same
  dummy_dat <- data.frame(date = NA,
                          depth_bin = factor('[0,2]', levels = levels(bias_dat$depth_bin)),
                          model_type = factor(rep(c('PB bias', 'DL bias', 'PGDL bias'), 2), levels = levels(bias_dat$model_type)),
                          median_bias = c(rep(bias_low, 3), rep(bias_high, 3)),
                          year = 2012,
                          doy = NA, stringsAsFactors = FALSE)

  # add in dummy dat
  bias_dat <- bind_rows(bias_dat, dummy_dat) %>%
    filter(!is.na(median_bias))


  return(bias_dat)

}

plot_bias_dat <- function(out_file, observations, lake, experiment, type) {
  plot_dat <- get_bias_dat(test_data = observations, lake, experiment, type)

  if (type == 'year') {
    date_labels <- c('J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D')
    date_breaks <- c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
    # dummy for hlines
    dummy_hline <- data.frame(model_type = rep(levels(plot_dat$model_type), 3),
                              year = rep(c(2012, 2016, 2017), 4),
                              int = rep(c(NA, 0, 0, 0), 3))
  } else {
    date_labels <- c('July', 'Aug', 'Sep')
    date_breaks <- c(182, 213, 244)
    # dummy for hlines
    dummy_hline <- data.frame(model_type = rep(levels(plot_dat$model_type), 3),
                              year = rep(2009:2017, 4),
                              int = rep(c(NA, 0, 0, 0), 3))
  }
  p <- ggplot(plot_dat, aes(x = doy, y = median_bias)) +
    geom_line(aes(group = depth_bin, color = depth_bin), alpha = 0.7) +
    geom_point(aes(group = depth_bin, color = depth_bin),
               alpha = 0.7, size = 0.3, fill = 'white') +
    geom_hline(data = dummy_hline, aes(yintercept = int), linetype = 2) +
    viridis::scale_color_viridis(discrete = TRUE, direction = -1) + #, palette = 'RdYlBu'
    #scale_shape_manual(values = c(21, 22, 23)) +
    scale_x_continuous(breaks = date_breaks,
                       labels = date_labels)+
    #facet_wrap(Depth_cat~year, ncol = 3, scales = 'free_y') +
    facet_grid(rows = vars(model_type), cols = vars(year), scales = 'free_y') +
    theme_bw()+
    theme(#strip.text = element_blank(),
      strip.background = element_blank(),
      legend.position = 'bottom',
      legend.direction="horizontal",
      panel.grid = element_blank(),
      #legend.background = element_rect(fill = 'white', color = 'black')
      legend.text = element_text(margin = margin(t = 0, b = 0))
    ) +
    labs( y = 'Temperature or Bias (deg C)', x = '') +
    guides(color = guide_legend(title = 'Depth (m)', title.position = 'left', ncol = 12,
                                label.position = 'left', direction = "horizontal"))

  ggsave(filename = out_file, plot = p, height = 7, width = 10, units = 'in')

}





