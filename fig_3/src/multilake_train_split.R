
calc_test_masks <- function(obs_data, site_id, test_location = 1/3){

  this_lakes_obs <- obs_data %>% filter(nhd_id == site_id) %>% arrange(desc(date))
  split_date <- this_lakes_obs %>% filter(row_number() <= (1-test_location) * n()) %>% tail(1) %>% pull(date)
  mask_end <- this_lakes_obs %>% filter(date <= split_date) %>% head(1) %>% pull(date)
  mask <- data.frame(experiment_detail= paste0('fig_3_', site_id), mask_start = min(this_lakes_obs$date), mask_end = mask_end, stringsAsFactors = FALSE)

  return(mask)
}

#' filepath ~ "{dir}/nhd_4250588_train_010_profiles_experiment_01.feather"
subset_training_random <- function(filepath, obs_data, test_mask){
  # create the seed based on the filepath
  # determine profile number based on the filepath
  name_details <- basename(filepath) %>% strsplit('[_]') %>% .[[1]]
  exp_n <- name_details[7] %>% strsplit('[.]') %>% .[[1]] %>% head(1) %>% as.numeric()
  prof_n <- name_details[4] %>% as.numeric()
  site_id <- paste0('nhd_', name_details[2])
  random_seed <- exp_n + as.numeric(name_details[2]) + prof_n# combo of experiment number, number of profiles, and nhd_id as a numeric

  non_test_obs <- obs_data %>% filter(nhd_id == site_id) %>% arrange(desc(date)) %>% filter(date < test_mask$mask_start | date > test_mask$mask_end) %>%
    group_by(nhd_id, date, depth) %>% summarize(temp = mean(temp, na.rm = TRUE))

  set.seed(random_seed)
  train_dates <- non_test_obs %>% pull(date) %>% unique() %>% sample(size = prof_n, replace = FALSE)

  non_test_obs %>% filter(date %in% train_dates) %>% arrange(date) %>%
    select(DateTime = date, Depth = depth, temp) %>%
    readr::write_csv(filepath)
}

subset_training <- function(filepath, obs_data, test_mask){
  # create the seed based on the filepath
  # determine profile number based on the filepath
  name_details <- basename(filepath) %>% strsplit('[_]') %>% .[[1]]
  site_id <- paste0('nhd_', name_details[2])

  obs_data %>% filter(nhd_id == site_id) %>% arrange(desc(date))  %>% filter(date < test_mask$mask_start | date > test_mask$mask_end) %>% arrange(date) %>%
    group_by(nhd_id, date, depth) %>% summarize(temp = mean(temp, na.rm = TRUE)) %>%
    select(DateTime = date, Depth = depth, temp) %>%
    readr::write_csv(filepath)
}

subset_testing <- function(filepath, obs_data, test_mask){
  name_details <- basename(filepath) %>% strsplit('[_]') %>% .[[1]]
  site_id <- paste0('nhd_', name_details[2])

  test_obs <- obs_data %>% filter(nhd_id == site_id) %>% arrange(desc(date)) %>% filter(date >= test_mask$mask_start & date <= test_mask$mask_end) %>% arrange(date) %>% arrange(date) %>%
    group_by(nhd_id, date, depth) %>% summarize(temp = mean(temp, na.rm = TRUE)) %>%
    select(DateTime = date, Depth = depth, temp) %>%
    readr::write_csv(filepath)
}


create_masks_feather <- function(filepath, ...){
  masks_in <- list(...)
  masks_out <- Reduce(function(x, y) merge(x, y, all=TRUE), masks_in)
  feather::write_feather(masks_out, path = filepath)
}
