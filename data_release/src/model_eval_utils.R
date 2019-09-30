calculate_RMSE <- function(target_name, test_file, ...){
  test_data <- read_csv(test_file)
  # test data has a field for "exper_n" which is the experiment number the test corresponds to `experiment_0{n}` in the prediction output files
  # test data also has a field for "exper_id" which lines up with things like "season" "similar" or "year" from the prediction file name
  # each "predict_file" will have an RMSE, an exper_n, an exper_id, a exper_model ("pgdl","dl",or "pb") and other stats like num dropped and num used

  predict_files <- c(...)

  data <- purrr::map(predict_files, function(x) {
    # "out/sp_season_predict_pb.csv":
    file_splits <- basename(x) %>% strsplit('[_]') %>% .[[1]]
    exper_model <- tail(file_splits, 1) %>%  strsplit('[.]') %>% .[[1]] %>% head(1)
    predictions <- read_csv(x)
    # and group_by exper_n and exper_id before doing this...?
    # gather(depth_code, temp, -DateTime) %>%
    #   mutate(Depth = as.numeric(substring(depth_code, 6))) %>%
    #   select(DateTime, Depth, temp) %>%
    browser()
    pred_obs <- bind_rows(lapply(unique(test_data$date), function(dt) {



      pred_1d <- filter(predictions, date == dt, !is.na(depth))

      obs_1d <- filter(test_data, date == dt) %>%
        rename(test_data = temp)

      tryCatch({
        if(nrow(pred_1d) == 0) stop(sprintf('no predictions on %s', dt))
        if(min(pred_1d$depth) != 0) warning(sprintf('no GLM prediction at 0m on %s', dt))
        mutate(obs_1d, pred = approx(x=pred_1d$depth, y=pred_1d$temp, xout=obs_1d$depth, rule=1)$y)
      }, error=function(e) {
        message(sprintf('approx failed for mendota on %s: %s', dt, e$message))
        mutate(obs_1d, pred = NA)
      })
    }))
    browser()
  }) %>% purrr::reduce(rbind)
  browser()
}
