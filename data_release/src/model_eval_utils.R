# function to read a test file and identify the corresponding predictions
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

as.rmse <- function(date, depth, pred, exper_n, exper_id, test_data = test_data){

  exper_n <- unique(exper_n)
  stopifnot(length(exper_n) == 1)

  exper_type <- unique(exper_id) %>% strsplit('[_]') %>% .[[1]] %>% head(1)

  filtered_test_data <- filter(test_data, exper_n == !!exper_n, exper_type == !!exper_type) %>%
    select(-exper_n, -exper_type)
  pred_obs <- prep_pred_obs(filtered_test_data, data.frame(date = date, depth = depth, pred = pred))

  sqrt(mean((pred_obs$pred - pred_obs$obs)^2, na.rm=TRUE))

}
as.hist_rmse <- function(date, depth, pred, site_id, exper_id, test_data = test_data){

  site_id <- unique(site_id)
  stopifnot(length(site_id) == 1)

  exper_type <- unique(exper_id) %>% strsplit('[_]') %>% .[[1]] %>% head(1)

  filtered_test_data <- filter(test_data, site_id == !!site_id, exper_type == !!exper_type) %>%
    select(-site_id, -exper_type)
  pred_obs <- prep_pred_obs(filtered_test_data, data.frame(date = date, depth = depth, pred = pred))

  sqrt(mean((pred_obs$pred - pred_obs$obs)^2, na.rm=TRUE))

}

calculate_hist_RMSE <- function(filename, test_file, pred_ind){
  test_data <- read_csv(test_file)
  # test data have "site_id"    "exper_type"
  predict_files <- yaml.load_file(pred_ind) %>% names


  # chunk predict files because holding all of the predictions in memory is too much.
  file_chunk_size <- 30

  start_idx <- seq(1, to = length(predict_files), by = 50)
  end_idx <- c(tail(start_idx, -1) - 1, length(predict_files))


  rmse_out <- data.frame(exper_n = c(), exper_id = c(), site_id = c(), model_type = c(), rmse = c())

  for (i in 1:length(start_idx)){
    these_files <- predict_files[start_idx[i]:end_idx[i]]
    data <- purrr::map(these_files, function(x) {
      # "out/nhd_2647582_predict_pgdl.csv":
      file_splits <- basename(x) %>% strsplit('[_]') %>% .[[1]]
      site_id <- head(file_splits, 2) %>% paste(collapse = '_')
      model_type <- tail(file_splits, 1) %>% strsplit('[.]') %>% .[[1]] %>% .[1]

      read_csv(x, col_types = cols(
        .default = col_double(),
        date = col_date(format = ""),
        exper_n = col_integer(),
        exper_id = col_character())
      ) %>%
        gather(depth_code, temp, -date, -exper_n, -exper_id) %>%
        mutate(depth = as.numeric(substring(depth_code, 6)), site_id = site_id, model_type = model_type) %>%
        select(date, depth, pred = temp, exper_n, exper_id, site_id, model_type) %>% arrange(date)
    }) %>% purrr::reduce(rbind)

    rmse <- data %>% group_by(exper_n, exper_id, site_id, model_type) %>%
      summarize(rmse = suppressMessages(as.hist_rmse(date, depth, pred, site_id, exper_id, test_data = test_data)))
    message('done with files', start_idx[i], 'to', end_idx[i])
    if (i == 1){
      rmse_out <- rmse
    } else{
      rmse_out <- rbind(rmse_out, rmse)
    }

  }



  write_csv(rmse_out, path = filename)
}


calculate_RMSE <- function(filename, test_file, ...){
  test_data <- read_csv(test_file)
  # test data has a field for "exper_n" which is the experiment number the test corresponds to `experiment_0{n}` in the prediction output files
  # test data also has a field for "exper_typ" which lines up with things like "season" "similar" or "year" from the prediction file name
  # each "predict_file" will have an RMSE, an exper_n, an exper_id, a model_type ("pgdl","dl",or "pb") and other stats like num dropped and num used

  predict_files <- c(...)

  data <- purrr::map(predict_files, function(x) {
    # "out/sp_season_predict_pb.csv":
    file_splits <- basename(x) %>% strsplit('[_]') %>% .[[1]]
    model_type <- tail(file_splits, 1) %>%  strsplit('[.]') %>% .[[1]] %>% head(1)

    read_csv(x) %>%
      gather(depth_code, temp, -date, -exper_n, -exper_id) %>%
      mutate(depth = as.numeric(substring(depth_code, 6)), model_type = model_type) %>%
      select(date, depth, pred = temp, exper_n, exper_id, model_type) %>% arrange(date)
  }) %>% purrr::reduce(rbind)

  rmse <- data %>% group_by(exper_n, exper_id, model_type) %>%
    summarize(rmse = as.rmse(date, depth, pred, exper_n, exper_id, test_data = test_data))

  write_csv(rmse, path = filename)
}
