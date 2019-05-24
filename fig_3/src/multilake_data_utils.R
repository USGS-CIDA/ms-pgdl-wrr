filter_temperature_obs <- function(obs, min_dates, min_per_date, strat_threshold, min_date){

  initial_cut <- filter(obs, !(nhd_id == 'nhd_13344284' & date > as.Date('2012-04-01') & date < as.Date('2013-12-31'))) %>%
    group_by(nhd_id, date) %>% summarise(n_obs = length(temp), obs_range = diff(range(temp))) %>%
    filter(n_obs >= min_per_date, date >= as.Date(min_date))


  save_ids <- initial_cut %>% group_by(nhd_id) %>%
    summarize(perc_strat = sum(obs_range > 1)/length(obs_range), n_dates = length(n_obs)) %>%
    filter(n_dates >= min_dates & perc_strat > strat_threshold) %>% pull(nhd_id)

  initial_cut %>% filter(nhd_id %in% save_ids) %>%
    left_join(y = obs, by = c('nhd_id','date')) %>% select(-n_obs, -obs_range)
}

filter_model_lakes <- function(obs, skip_ids){
  filter(obs, !nhd_id %in% skip_ids) %>% pull(nhd_id)
}




export_test_train <- function(filepath, site_id, obs, num_train_dates){
  this_lake <- filter(obs, nhd_id == site_id)
  un_dates <-  unique(this_lake$date)

  set.seed(42)
  train_dates <- sample(un_dates, size = num_train_dates, replace = FALSE)
  this_lake %>% mutate(training_obs = date %in% train_dates) %>%
    write_feather(path = filepath)
}

