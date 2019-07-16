calc_pretrainer_RMSE <- function(){
  # manual download from https://drive.google.com/drive/folders/188tY64FV9dS8nGyx_QFfGRJSclGjcCiL
  files <- dir('~/Downloads/pre-training data/')
  nhd_ids <- sapply(files, FUN = function(x) strsplit(x, '[_]')[[1]][2], USE.NAMES = FALSE)
  pred_obs <- bind_rows(lapply(nhd_ids, prep_pred_obs))
  compute_RMSEs(pred_obs) %>%
    readr::write_tsv('fig_3/in/glm_uncal_rmses.tsv')
}

compute_RMSEs <- function(pred_obs) {

  results <- pred_obs %>%
    group_by(nhd_id) %>%
    summarize(
      obs_removed = length(which(is.na(pred))),
      shallowest_removed = if(any(is.na(pred))) min(depth[is.na(pred)]) else NA,
      deepest_removed = if(any(is.na(pred))) max(depth[is.na(pred)]) else NA,
      rmse = if(all(is.na(pred))) NA else sqrt(mean((pred - obs)^2, na.rm=TRUE)))

  return(results)
}

prep_pred_obs <- function(nhd_id='10596466') {

  message(nhd_id)

  nhdid <- nhd_id
  data_path <- "fig_3/yeti_sync"

  obs <- readr::read_csv(sprintf('%s/nhd_%s_test_all_profiles.csv', data_path, nhd_id)) %>%
    select(date = DateTime, depth = Depth, temp)

  glm_preds <- feather::read_feather(sprintf('%s/nhd_%s_temperatures.feather', '~/Downloads/pre-training data', nhd_id)) %>%
    mutate(date = as.Date(DateTime)) %>%
    select(-DateTime, -ice) %>%
    gather(depth_code, temp, -date) %>%
    mutate(depth = as.numeric(substring(depth_code, 6))) %>%
    select(date, depth, temp) %>%
    filter(date %in% obs$date)

  pred_obs <- bind_rows(lapply(unique(obs$date), function(dt) {
    pred_1d <- filter(glm_preds, date == dt)

    obs_1d <- filter(obs, date == dt) %>%
      rename(obs = temp)

    tryCatch({
      if(nrow(pred_1d) == 0) stop(sprintf('no predictions on %s', dt))
      if(min(pred_1d$depth) != 0) warning(sprintf('no GLM prediction at 0m on %s', dt))
      mutate(obs_1d, pred = approx(x=pred_1d$depth, y=pred_1d$temp, xout=depth, rule=1)$y)
    }, error=function(e) {
      message(sprintf('approx failed for %s on %s: %s', nhd_id, dt, e$message))
      mutate(obs_1d, pred = NA)
    })
  }))

  pred_obs %>%
    mutate(nhd_id = nhd_id)
}

plot_calibrated_figure_3 <- function(){

  library(dplyr)
  library(readr)
  library(tidyr)



  # PGDL_all = as.numeric(`PGDL(<1 RMSE)`), DL_all = as.numeric(`DL (< 1 RMSE)`),
  # PGDL_10 = as.numeric(`PGDL_10(< 1 RMSE)`), DL_10 = as.numeric(`DL_10(< 1 RMSE)`),

  # PGDL_all = as.numeric(`PGDL(< .7 RMSE)`), DL_all = as.numeric(`DL (< .7 RMSE)`),
  # PGDL_10 = as.numeric(`PGDL_10 ( < .7 RMSE)`), DL_10 = as.numeric(`DL(< .7 RMSE)`), # typo in spreadsheet!!


  # PGDL_all = as.numeric(`PGDL(400 ep)`), DL_all = as.numeric(`DL(400 ep)`),
  # PGDL_10 = as.numeric(`PGDL_10(400 ep)`), DL_10 = as.numeric(`DL_10(400 ep)`),

  jared_pgdl <- readr::read_csv('~/Downloads/Comparing Stopping Conditions - Sheet1.csv')  %>% #'fig_3/in/glm_uncal_vs_PGDL_rmses.csv'
    mutate(PGDL_all = as.numeric(`PGDL(400 ep)`), DL_all = as.numeric(`DL(400 ep)`),
           PGDL_10 = as.numeric(`PGDL_10(400 ep)`), DL_10 = as.numeric(`DL_10(400 ep)`),
           PB_uncal = `GLM uncal rmse`,
           n_train = as.numeric(`train days`),
           nhd_id = paste0('nhd_', nhd_id)) %>% slice(1:68) %>%
    select(nhd_id, PGDL_10, DL_10, PGDL_all, DL_all, n_train)


  glm_rmse <- readr::read_tsv('fig_3/in/glm_uncal_rmses.tsv') %>%
    rename(PB_uncal = rmse) %>%
    mutate(nhd_id = paste0('nhd_', as.character(nhd_id))) %>%
    select(nhd_id, PB_uncal)

  plot_data <- left_join(jared_pgdl, glm_rmse, by = "nhd_id")

  optim_files <- dir('yeti_in/fig_3/')
  cal_GLM <- data.frame(nhd_id = c(), PB_cal = c(), exper = c())
  for (file in optim_files){
    this_data <- read_csv(file.path('yeti_in/fig_3/', file), col_types = 'cddddd') %>%
      mutate(exper = paste0(strsplit(file, '[_]')[[1]][3], '_', strsplit(file, '[_]')[[1]][6])) %>% select(nhd_id, PB_cal = test_rmse, exper)
    cal_GLM <- rbind(cal_GLM, this_data)
  }

  cal_GLM <- cal_GLM %>% spread(exper, PB_cal) %>%
    rename(PB_10_01 = `010_01`, PB_10_02 = `010_02`, PB_10_03 = `010_03`, PB_all = all_NA) %>%
    rowwise() %>% mutate(PB_10 = median(c(PB_10_01, PB_10_02, PB_10_03), na.rm = TRUE)) %>% select(nhd_id, PB_10, PB_all)
  plot_data <- left_join(plot_data, cal_GLM, by = 'nhd_id') %>% filter(!is.na(PB_all), !is.na(PB_10), !is.na(PGDL_all))

  plot_data <- left_join(plot_data, readr::read_csv('~/Downloads/site_count_68lakes.csv'))
  n_sims <- nrow(plot_data)

  sprintf('Predictions from PGDL models applied to %s lakes were more accurate or as accurate (within +/-0.05°C RMSE) as all but %s of the calibrated PB models and %s DL models (',
          n_sims,
          sum(plot_data$PGDL_all > plot_data$PB_all+0.05),
          sum(plot_data$PGDL_all > plot_data$DL_all+0.05)) %>% message()

  sprintf('The median RMSE (across all lakes) was %s° for the PGDL model, %s°C for PB, and %s°C for DL. ',
          round(median(plot_data[['PGDL_all']]),2),
          round(median(plot_data[['PB_all']]),2),
          round(median(plot_data[['DL_all']]),2)) %>% message()

  sprintf('The range of prediction accuracy for PGDL models was %s to %s°C, %s to %s°C for PB, and %s to %s°C for DL. ',
          round(range(plot_data$PGDL_all)[1], 2), round(range(plot_data$PGDL_all)[2], 2),
          round(range(plot_data$PB_all)[1], 2), round(range(plot_data$PB_all)[2], 2),
          round(range(plot_data$DL_all)[1], 2), round(range(plot_data$DL_all)[2], 2)) %>%
    message()

  sprintf('The uncalibrated GLM predictions (which are used for pre-training PGDL) had a median RMSE of %s°C, with a range of %s to %s°C.',
          round(median(plot_data[['PB_uncal']]),2),
          round(range(plot_data[['PB_uncal']])[1],2),
          round(range(plot_data[['PB_uncal']])[2],2)) %>%
    message()

  sprintf('The uncalibrated GLM predictions were worse than PB in %s lakes (%s%% of total)',
          sum(plot_data$PB_uncal > plot_data$PB_all),
          round((sum(plot_data$PB_uncal > plot_data$PB_all)/n_sims)*100, 1)) %>%
    message()

  sprintf('were variable across lakes, with %s lakes improving RMSE by over 2° compared to pre-trainer RMSEs, while %s lakes had PGDL predictions that were approximately equal to the pre-trainer (range of %s to %s°C increase in RMSE). ',
          sum(plot_data$PGDL_all + 2.0 < plot_data$PB_uncal),
          sum(plot_data$PGDL_all > plot_data$PB_uncal + 0.05 | plot_data$PGDL_all > plot_data$PB_uncal - 0.05),
          round(range(filter(plot_data, PGDL_all - PB_uncal > 0 ) %>% mutate(diff = PGDL_all - PB_uncal) %>% pull(diff))[1],3),
          round(range(filter(plot_data, PGDL_all - PB_uncal > 0 ) %>% mutate(diff = PGDL_all - PB_uncal) %>% pull(diff))[2],3)) %>%
    message()

  sprintf('The difference in RMSE between PB and PGDL ranged from %s to %s°C and %s to %s°C for DL to PGDL',
         round(range(plot_data$PB_all - plot_data$PGDL_all)[1], 2), round(range(plot_data$PB_all - plot_data$PGDL_all)[2], 2),
         round(range(plot_data$DL_all - plot_data$PGDL_all)[1], 2), round(range(plot_data$DL_all - plot_data$PGDL_all)[2], 2)) %>%
    message()

  sprintf('When observations were artificially removed to leave only 10 dates for training, predictions from PGDL models were more accurate than %s of the calibrated PB models (%s%% of total) and more accurate than %s DL models (Figure 4).',
          sum(plot_data$PB_10 > plot_data$PGDL_10 - 0.05),
          round((sum(plot_data$PB_10 > plot_data$PGDL_10 - 0.05)/n_sims)*100, 1),
          ifelse(all(plot_data$DL_10 > plot_data$PGDL_10 - 0.05), 'all', sum(plot_data$DL_10 > plot_data$PGDL_10 - 0.05))) %>% message()

  sprintf('The median RMSE (across all lakes) for 10 training profiles was %s° for PGDL, %s°C for PB, and %s°C for DL. ',
          round(median(plot_data[['PGDL_10']]),2),
          round(median(plot_data[['PB_10']]),2),
          round(median(plot_data[['DL_10']]),2)) %>% message()

  sprintf('With the reduced training dataset, %s lakes had worse PGDL predictions compared to the pre-trainer (range of %s to %s°C increase in RMSE) while %s lakes with PGDL models decreased prediction RMSE by more than 1° compared to the pre-trainer (greatest improvement was a reduction of %s°C RMSE). ',
          sum(plot_data$PGDL_10 > plot_data$PB_uncal),
          round(range(filter(plot_data, PGDL_10 - PB_uncal > 0 ) %>% mutate(diff = PGDL_10 - PB_uncal) %>% pull(diff))[1],3),
          round(range(filter(plot_data, PGDL_10 - PB_uncal > 0 ) %>% mutate(diff = PGDL_10 - PB_uncal) %>% pull(diff))[2],3),
          sum(plot_data$PGDL_10 + 1.0 < plot_data$PB_uncal),
          round(max(plot_data$PB_uncal- plot_data$PGDL_10), 2)) %>%
    message()

  png(filename = 'figures/figure_3_wrr.png', width = 7, height = 7, units = 'in', res = 200)

  par(omi = c(0.6,0,0.1,0.2), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.5, 0.5)
  xlim <- c(0.7, 4.5)

  positions <- list(PB_uncal = 1,
                    PB_10 = 0.8,
                    PB_all = 2,
                    DL_10 = 1.775,
                    DL_all = 3,
                    PGDL_10 = 2.775,
                    PGDL_all = 4)

  plot(NA, NA, xlim = xlim, ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)
  library(beanplot)
  bean_w <- 0.65
  plot_data <- plot_data %>% select(-nhd_id, -n_unique_sites, -n_train, -PB_10, -PGDL_10, -DL_10)

  beanplot(plot_data$PB_uncal, plot_data$PB_all, plot_data$DL_all, plot_data$PGDL_all, maxwidth = bean_w, what=c(0,1,0,0), log = "", add = TRUE,
            axes = F, border = NA, at = c(positions$PB_uncal, positions$PB_all, positions$DL_all, positions$PGDL_all), col = list('grey65','#1b9e77','#d95f02','#7570b3'))

  med_w <- 0.4
  ind_w <- 0.03
  for (x_bin in 1:ncol(plot_data)){
    mod_name <- names(plot_data)[x_bin]
    segments(x0 = positions[[mod_name]]-ind_w, x1 = positions[[mod_name]]+ind_w, y0 = plot_data[[mod_name]], col = 'black')
    #segments(x0 = positions[[mod_name]]-ind_w/2, x1 = positions[[mod_name]]+ind_w/2, y0 = plot_data[[mod_name]], col = 'white')
    segments(x0 = positions[[mod_name]]-med_w/2, x1 = positions[[mod_name]]+med_w/2, y0 = median(plot_data[[mod_name]]), col = 'black', lwd = 2)
  }


  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)

  # axis(1, at = c(-100, positions$PB_uncal, positions$PB_10, positions$PB_all, positions$DL_10, positions$DL_all, positions$PGDL_10, positions$PGDL_all, 1e10),
  #      labels = c("", expression("PB"['0']), expression("PB"['10']), expression("PB"['all']),
  #                 expression("DL"['10']), expression("DL"['all']),
  #                 expression("PGDL"['10']), expression("PGDL"['all']), ""), tck = -0.01)

  #par(mai = c(0,0,0,0), mgp = c(2, 1.5,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""),  tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0,0,0,0), mgp = c(2, 1.5,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), "Based", "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}
