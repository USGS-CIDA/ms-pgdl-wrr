library(readr)
library(feather)
library(dplyr)
library(tidyr)
library(ggplot2)

data_path <- 'fig_3/in'
doc_path <- 'fig_3/doc'

#### Prepare data ####

# Read Jared's record of the training and test date ranges
jared_results <- readr::read_tsv(file.path(data_path, 'glm_uncal_vs_PGDL_rmses.tsv'), n_max = 68, col_types='ciiddiDDDDiic')
jared_results[52, 'Train Begin'] <- as.Date('1995-07-24') # correct funny Excel formatting

# Get a list of IDs for lakes that were modeled
nhd_ids <- dir(data_path, pattern='.*\\_temperatures.feather') %>%
  gsub('nhd_', '', .) %>%
  gsub('_temperatures.feather', '', .)

prep_pred_obs <- function(nhd_id='10596466') {

  message(nhd_id)

  nhdid <- nhd_id
  test_dates <- jared_results %>%
    filter(nhd_id == as.numeric(nhdid))

  obs <- feather::read_feather(sprintf('%s/nhd_%s_test_train.feather', data_path, nhd_id)) %>%
    filter(date >= test_dates$`Test Begin`, date <= test_dates$`Test End`) %>%
    select(date, depth, temp)

  glm_preds <- feather::read_feather(sprintf('%s/nhd_%s_temperatures.feather', data_path, nhd_id)) %>%
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

pred_obs <- bind_rows(lapply(nhd_ids, prep_pred_obs))

#### Compute and summarize RMSEs ####

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

rmses <- compute_RMSEs(pred_obs)

rmses %>% arrange(desc(rmse)) %>%
  readr::write_tsv(x=., path=sprintf('%s/glm_uncal_rmses.tsv', doc_path))

ggplot(rmses, aes(x=rmse)) + geom_density(fill='cornflowerblue', color=NA) + geom_rug(color='cornflowerblue') +
  theme_bw() + xlab('RMSE') + ggtitle('RMSEs of uncalibrated GLM predictions')
ggsave(sprintf('%s/glm_uncal_rmses.png', doc_path), width=5, height=4)

#### Compute RMSEs at surface and at deepest depth

surface_rmse <- pred_obs %>%
  filter(depth == 0) %>%
  group_by(nhd_id) %>%
  summarize(
    rmse = if(all(is.na(pred))) NA else sqrt(mean((pred - obs)^2, na.rm=TRUE)))
deep_rmse <- pred_obs %>%
  filter(!is.na(pred)) %>%
  group_by(nhd_id, date) %>%
  filter(depth == max(depth)) %>%
  group_by(nhd_id) %>%
  summarize(
    rmse = if(all(is.na(pred))) NA else sqrt(mean((pred - obs)^2, na.rm=TRUE)))
summary(surface_rmse$rmse)
# Using old train-test split (from jordan's data)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 0.9492  1.4768  1.6022  1.7393  1.8880  3.9278
# Using new train-test split (from jared's dates):
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 0.7435  1.4395  1.6096  1.7290  2.0035  3.6818
summary(deep_rmse$rmse)
# Using old train-test split (from jordan's data)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.011   1.834   2.526   2.860   3.272   9.127
# Using new train-test split (from jared's dates):
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 1.108   1.881   2.504   2.812   3.331   7.804

#### Choose top-priority lakes for PGML ####

# pick 5 lakes with RMSE between 2 and 3, plus any criteria from jordan?

plot_pred_obs_depth <- function(rmse_row) {
  g <- pred_obs %>%
    filter(nhd_id == rmse_row$nhd_id) %>%
    ggplot(aes(x=obs, y=pred, color=depth)) +
    geom_point() +
    geom_abline() +
    theme_classic() +
    ggtitle(sprintf('Lake %s (RMSE = %0.2f)', rmse_row$nhd_id, rmse_row$rmse))
  ggsave(sprintf('%s/%s_pred_obs_depth.png', doc_path, rmse_row$nhd_id), g)
  g
}
plot_hilo_ts <- function(rmse_row) {
  dat <- pred_obs %>%
    filter(nhd_id == rmse_row$nhd_id)
  depth_counts <- dat %>%
    group_by(depth) %>%
    summarize(n_obs = n())
  decent_deep <- depth_counts %>%
    filter(n_obs > max(n_obs)/2) %>%
    filter(depth == max(depth)) %>%
    pull(depth)

  g <- dat %>%
    filter(depth %in% c(0, decent_deep)) %>%
    filter(!is.na(pred)) %>%
    ggplot(aes(x=date, group=depth, color=depth)) +
    geom_line(aes(y=pred)) +
    geom_point(aes(y=obs)) +
    theme_classic() +
    facet_grid(depth ~ .) +
    ylab('Preds (points) and obs (lines) for 2 depths') +
    ggtitle(sprintf('Lake %s (RMSE = %0.2f)', rmse_row$nhd_id, rmse_row$rmse))
  ggsave(sprintf('%s/%s_hilo_ts.png', doc_path, rmse_row$nhd_id), g)
  g
}
plot_heat <- function(rmse_row) {
  g <- pred_obs %>%
    filter(nhd_id == rmse_row$nhd_id) %>%
    ggplot(aes(x=date, y=depth)) +
    geom_point(aes(fill=pred-obs, color=pred-obs), size=3, alpha=0.8) +
    scale_color_gradient2() + scale_fill_gradient2() +
    theme(panel.grid = element_blank()) +
    scale_y_reverse() +
    ggtitle(sprintf('Lake %s (RMSE = %0.2f)', rmse_row$nhd_id, rmse_row$rmse))
  ggsave(sprintf('%s/%s_heat.png', doc_path, rmse_row$nhd_id), g)
  g
}

rmse <- rmses[which.min(abs(rmses$rmse - 2)),] # 13343906
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)

rmse <- rmses[which.min(abs(rmses$rmse - 2.5)),] # 2349188
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)

rmse <- rmses[which.min(abs(rmses$rmse - median(rmses$rmse))),] # 1099052
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)

rmse <- rmses[which.min(abs(rmses$rmse - 3)),] # 120052351
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)

rmse <- rmses[which.min(abs(rmses$rmse - 4)),] # 1099240
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)

rmse <- rmses[which.min(abs(rmses$rmse - 5)),] # 1101590
plot_pred_obs_depth(rmse)
plot_hilo_ts(rmse)
plot_heat(rmse)
