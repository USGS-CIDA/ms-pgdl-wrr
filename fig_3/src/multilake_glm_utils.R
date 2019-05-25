run_optim_glm <- function(driver_file, nml_file, train_file, test_file, sheets_id){

  sim_dir <- '.sim_raw'
  on.exit(unlink(sim_dir, recursive = TRUE))

  results_df <- optim_multilake_glm(driver_file, nml_file, train_file, test_file, sim_dir)
  post_to_results_sheets(results_df, sheets_id)


  return(results_df)
}

post_to_results_sheets <- function(results_df, sheets_id){
  gs_anchor <- gs_read(sheets_id) %>% nrow %>% .+1 %>% sprintf("A%s", .)
  gs_edit_cells(sheets_id, ws = 1, input = results_df, anchor = gs_anchor, byrow = TRUE)
}


optim_multilake_glm <- function(driver_file, nml_file, train_file, test_file, sim_dir){

  nhd_id <- basename(driver_file) %>% strsplit('[_]') %>% .[[1]] %>% .[2] %>% sprintf('nhd_%s', .)
  dir.create(sim_dir)
  nml_obj <- read_nml(nml_file)
  cd_start <- get_nml_value(nml_obj, "cd")
  kw_start <- get_nml_value(nml_obj, "Kw")
  meteo_file <- basename(driver_file)

  # set all fixed nml values
  # check metweo file is named right in nml??
  file.copy(from = driver_file, to = sprintf('%s/%s', sim_dir, meteo_file))
  nml_obj <- set_nml(nml_obj, arg_list = list(meteo_fl = meteo_file))

  train_filepath <- sprintf('%s/train_data.csv', sim_dir)
  test_filepath <- sprintf('%s/test_data.csv', sim_dir)
  file.copy(from = train_file, to = train_filepath)
  file.copy(from = test_file, to = test_filepath)

  initial_params = c('cd'=cd_start, coef_wind_stir=0.23, Kw = kw_start)
  parscale = c('cd'=0.0001, coef_wind_stir=0.001, Kw = 0.1*kw_start)

  # optimize initial parameters
  out = optim(fn = run_cal_simulation, par=initial_params, control=list(parscale=parscale),
              train_csv_file = train_filepath, sim_dir = sim_dir, nml_obj = nml_obj)
  results <- data.frame(as.list(out$par), train_rmse = out$value)
  results$test_rmse <- compare_to_field(sprintf('%s/output.nc', sim_dir),
                                        test_filepath,
                                        metric = 'water.temperature')
  resutls$nhd_id <- nhd_id
  unlink(sim_dir, recursive = TRUE)

  return(results)
}



run_cal_simulation <- function(par, train_filepath, sim_dir, nml_obj){

  nml_path <- paste0(sim_dir,'/glm2.nml')
  nml_obj <- set_nml(nml_obj, arg_list = as.list(par)) # custom param shifts
  write_nml(glm_nml = nml_obj, file = nml_path)

  rmse = tryCatch({
    stop('forcing failure to test system')
    sim = run_glm(sim_dir, verbose = FALSE)
    last_time <- glmtools::get_var(sprintf('%s/output.nc', sim_dir), 'wind') %>%
      tail(1) %>% pull(DateTime)
    if (last_time < as.Date(as.Date(get_nml_value(nml_obj, "stop")))){
      stop('incomplete sim, ended on ', last_time)
    }
    rmse = compare_to_field(file.path(sim_dir, 'output.nc'),
                            field_file = train_filepath,
                            metric = 'water.temperature')
  }, error = function(e){
    message(e$message)
    return(10) # a high RMSE value
  })


  # need to compare to obs and return NLL or RMSE stats; using rmse for now

  message(rmse)

  return(rmse)
}
