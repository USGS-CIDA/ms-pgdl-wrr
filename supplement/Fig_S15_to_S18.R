# function to read a test file and identify the corresponding predictions
prep_pred_obs <- function(test_file, model_preds) {

  # read in the test observations
  obs <- feather::read_feather(test_file)

  # match up preds to obs, interpolating GLM predictions to match the observation depths
  pred_obs <- bind_rows(lapply(unique(obs$DateTime), function(dt) {
    pred_1d <- filter(model_preds, DateTime == dt, !is.na(Depth))

    obs_1d <- filter(obs, DateTime == dt) %>%
      rename(obs = temp)

    tryCatch({
      if(nrow(pred_1d) == 0) stop(sprintf('no predictions on %s', dt))
      if(min(pred_1d$Depth) != 0) warning(sprintf('no GLM prediction at 0m on %s', dt))
      mutate(obs_1d, pred = approx(x=pred_1d$Depth, y=pred_1d$temp, xout=obs_1d$Depth, rule=1)$y)
    }, error=function(e) {
      message(sprintf('approx failed for mendota on %s: %s', dt, e$message))
      mutate(obs_1d, pred = NA)
    })
  }))


  return(pred_obs)
}

get_timeseries <- function(lake, type, exp){

  library(dplyr)
  library(reticulate)
  library(readr)
  library(feather)
  library(stringr)
  library(tidyr)
  np <- import('numpy')


#lake in c('sparkling','mendota'))
#(type in c('year','season'))
 #model in c('PGRNN','RNN'))

  # get PGRNN (PGDL) output
  pgdl_dat <- RcppCNPy::npyLoad(sprintf('/Users/jread/Downloads/outputs_new/PGRNN_%s_%s_exp%s.npy', type, lake, exp)) %>%
    t() %>%
    as.data.frame() %>% setNames(paste0('temp_',seq(0, length.out = ifelse(lake == "mendota", 50, 37), by = 0.5))) %>% #length.out = 50 for Mendota
    mutate(DateTime = seq(as.Date("2009-04-02"), length.out = 3185, by = 'days')) %>%
    gather(depth_code, temp, -DateTime) %>%
    mutate(Depth = as.numeric(substring(depth_code, 6))) %>%
    select(DateTime, Depth, temp) %>%
    prep_pred_obs(sprintf('/Users/jread/Downloads/%s_%s_test.feather', lake, type), .) %>%
    rename(pgdl_pred = pred) %>%
    distinct()

  # get RNN (DL) output
  dl_dat <- RcppCNPy::npyLoad(sprintf('/Users/jread/Downloads/outputs/RNN_%s_%s_exp%s.npy', type, lake, exp)) %>%
    t() %>%
    as.data.frame() %>% setNames(paste0('temp_',seq(0, length.out = ifelse(lake == "mendota", 50, 37), by = 0.5))) %>% #length.out = 50 for Mendota
    mutate(DateTime = seq(as.Date("2009-04-02"), length.out = 3185, by = 'days')) %>%
    gather(depth_code, temp, -DateTime) %>%
    mutate(Depth = as.numeric(substring(depth_code, 6))) %>%
    select(DateTime, Depth, temp) %>%
    prep_pred_obs(sprintf('/Users/jread/Downloads/%s_%s_test.feather', lake, type), .) %>%
    rename(dl_pred = pred) %>%
    distinct()

  glm_preds <- feather::read_feather(sprintf('../lake_modeling/data_imports/out/%s_%s_calibrated_experiment_0%s.feather', lake, type, exp)) %>%
    mutate(DateTime = as.Date(DateTime)) %>%
    gather(depth_code, temp, -DateTime) %>%
    mutate(Depth = as.numeric(substring(depth_code, 6))) %>%
    select(DateTime, Depth, temp) %>%
    prep_pred_obs(sprintf('~/Downloads/%s_%s_test.feather', lake, type), .) %>%
    rename(glm_pred = pred) %>%
    distinct()

  all_dat <- left_join(pgdl_dat, dl_dat) %>%
    left_join(glm_preds) %>%
    filter(!is.na(obs)) %>% filter(!is.na(dl_pred), !is.na(pgdl_pred))

  return(all_dat)

}

plot_timeseries <- function(lake, type, exp) {
  library(dplyr)
  library(ggplot2)
  library(tidyr)

  ts_dat <- get_timeseries(lake, type, exp)
  #head(sparkling_year_exp1)

  depth_max <- ifelse(lake == 'sparkling', 18, 20)

  ts_dat_bias <- ts_dat %>%
    mutate(pgdl_bias = pgdl_pred - obs,
           dl_bias = dl_pred - obs,
           glm_bias = glm_pred - obs) %>%
    mutate(depth_bin = cut(Depth, breaks = seq(0, depth_max, by = 2),
                           include.lowest = TRUE)) %>%
    filter(!is.na(depth_bin)) %>%  # exclude Mendota depths that are >depth_max (20m)
    group_by(DateTime, depth_bin) %>%
    summarize(`Observed` = median(obs, na.rm = TRUE),
              `PB bias` = median(glm_bias, na.rm = TRUE),
              `PGDL bias` = median(pgdl_bias, na.rm = TRUE),
              `DL bias` = median(dl_bias, na.rm = TRUE))


  ts_dat_long <- gather(ts_dat_bias, key = 'variable', value = 'value', `Observed`, `PB bias`, `PGDL bias`, `DL bias`) %>%
    mutate(year = lubridate::year(DateTime),
           doy = lubridate::yday(DateTime))

  if (type == 'season') {
    ts_dat_long <- filter(ts_dat_long, year %in% c(2012, 2016, 2017))
  }
  # find min and max for bias scales
  bias_scales <- filter(ts_dat_long, variable != 'Observed')
  bias_range <- range(bias_scales$value, na.rm = TRUE)
  bias_low <- floor(bias_range[1]) - 1
  bias_high <- ceiling(bias_range[2]) + 1

  # add dummy obs to get scales the same
  dummy_dat <- data.frame(DateTime = NA,
                          depth_bin = factor('[0,2]', levels = levels(ts_dat_long$depth_bin)),
                          variable = rep(c('PB bias', 'DL bias', 'PGDL bias'), 2),
                          value = c(rep(bias_low, 3), rep(bias_high, 3)),
                          year = 2012,
                          doy = NA)

  # add in dummy dat
  ts_dat_long <- bind_rows(ts_dat_long, dummy_dat)

  ts_dat_long$variable <- factor(ts_dat_long$variable, levels = c('Observed', 'PB bias', 'DL bias', 'PGDL bias'))
  #  filter(lubridate::year(DateTime) == 2016)

  # create depth bins

  #sparkling_1m_18m <- filter(sparkling_year_exp1_long, Depth == 1|Depth == 18)
  #sparkling_1m_18m$Depth_cat <- paste0('Depth = ', sparkling_1m_18m$Depth, 'm')
  #sparkling_1m_18m$Depth_cat <- factor(sparkling_1m_18m$Depth_cat, levels = c('Depth = 1m', 'Depth = 18m'))
  # sparkling_1m_18m <- sparkling_1m_18m %>%
  #   mutate(variable = case_when(
  #     variable == 'dl_pred' ~ 'DL Predicted',
  #     variable == 'pgdl_pred' ~ 'PGDL Predicted',
  #     variable == 'obs' ~ 'Observed'
  #   ))
  #sparkling_1m_18m$variable <- factor(sparkling_1m_18m$variable, levels = c('Observed', 'DL Predicted', 'PGDL Predicted'))
  # hack for getting scales of bias figs the same


  # dummy for hlines
  dummy_hline <- data.frame(variable = rep(unique(ts_dat_long$variable), 3),
                            year = rep(c(2012, 2016, 2017), 4),
                            int = rep(c(NA, 0, 0, 0), 3))

  if (type == 'year') {
    date_labels <- c('Jan', '', 'Mar', '', 'May', '', 'July', '', 'Sep', '', 'Nov', '')
    date_breaks <- c(1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)
  } else {
    date_labels <- c('July', 'Aug', 'Sep')
    date_breaks <- c(182, 213, 244)
  }

  p <- ggplot(ts_dat_long, aes(x = doy, y = value)) +
    geom_line(aes(group = depth_bin, color = depth_bin), alpha = 0.7) +
    geom_point(aes(group = depth_bin, color = depth_bin),
               alpha = 0.7, size = 1, fill = 'white') +
    geom_hline(data = dummy_hline, aes(yintercept = int), linetype = 2) +
    viridis::scale_color_viridis(discrete = TRUE, direction = -1) + #, palette = 'RdYlBu'
    #scale_shape_manual(values = c(21, 22, 23)) +
    scale_x_continuous(breaks = date_breaks,
                       labels = date_labels)+
    #facet_wrap(Depth_cat~year, ncol = 3, scales = 'free_y') +
    facet_grid(rows = vars(variable), cols = vars(year), scales = 'free_y') +
    theme_bw()+
    theme(#strip.text = element_blank(),
      strip.background = element_blank(),
      #legend.position = c(0.15, 0.85),
      legend.position = 'right',
      #legend.margin = element_text(margin = margin(0,0,0,0)),
      #axis.title.x = element_blank(),
      panel.grid = element_blank(),
      #legend.background = element_rect(fill = 'white', color = 'black')
      legend.text = element_text(margin = margin(t = 0))
      ) +
    labs( y = 'Temperature or Bias (deg C)', x = '') +
    guides(color = guide_legend(title = 'Depth (m)', title.position = 'top', ncol = 1,
                                label.position = 'left'))

  ggsave(filename = sprintf('../lake_modeling/data_imports/figures/supp_fig_timeseries_%s_%s.png', lake, type),
         plot = p, height = 190, width = 230, units = 'mm')

}
#
# plot_timeseries(lake = 'sparkling', type = 'year', exp = 1)
# plot_timeseries(lake = 'sparkling', type = 'season', exp = 1)
#
# plot_timeseries(lake = 'mendota', type = 'year', exp = 1)
# plot_timeseries(lake = 'mendota', type = 'season', exp = 1)
