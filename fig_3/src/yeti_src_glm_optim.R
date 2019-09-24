run_optim_glm <- function(driver_file, nml_file, train_file, test_file, exper_id, sim_dir, glm_nix_dir){

  library(glmtools)
  library(dplyr)

  on.exit(unlink(sim_dir, recursive = TRUE))

  optim_multilake_glm(driver_file, nml_file, train_file, test_file, sim_dir, glm_nix_dir) %>%
    select(nhd_id, train_rmse, test_rmse, everything()) %>%
    write_csv(paste0('out/fig_3/', exper_id, "_results.csv"))
}

optim_multilake_glm <- function(driver_file, nml_file, train_file, test_file, sim_dir, glm_nix_dir){

  nhd_id <- basename(driver_file) %>% strsplit('[_]') %>% .[[1]] %>% .[2] %>% sprintf('nhd_%s', .)
  nml_obj <- read_nml(nml_file)
  cd_start <- get_nml_value(nml_obj, "cd")
  kw_start <- get_nml_value(nml_obj, "Kw")
  meteo_file <- basename(driver_file)

  results_log <- file.path(sim_dir, 'results_log.txt')
  file.create(results_log)

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
  min_rmse <<- 10
  # optimize initial parameters
  out = optim(fn = run_cal_simulation, par=initial_params, control=list(parscale=parscale),
              train_filepath = train_filepath, sim_dir = sim_dir, nml_obj = nml_obj, results_log = results_log, glm_nix_dir = glm_nix_dir)
  results <- data.frame(as.list(out$par), train_rmse = out$value)
  results$test_rmse <- compare_to_field(sprintf('%s/output.nc', sim_dir),
                                        test_filepath,
                                        metric = 'water.temperature')
  results$nhd_id <- nhd_id
  unlink(sim_dir, recursive = TRUE)

  return(results)
}

run_cal_simulation <- function(par, train_filepath, sim_dir, nml_obj, results_log, glm_nix_dir){

  nml_path <- file.path(sim_dir,'glm2.nml')

  nml_obj <- set_nml(nml_obj, arg_list = as.list(par)) # custom param shifts
  write_nml(glm_nml = nml_obj, file = nml_path)

  rmse = tryCatch({

    sim = run_glm_copy(sim_dir, glm_nix_dir)
    last_time <- glmtools::get_var(sprintf('%s/output.nc', sim_dir), 'wind') %>%
      tail(1) %>% pull(DateTime)
    if (last_time < as.Date(as.Date(get_nml_value(nml_obj, "stop")))){
      stop('incomplete sim, ended on ', last_time)
    }
    rmse = compare_to_field(file.path(sim_dir, 'output.nc'),
                            field_file = train_filepath,
                            metric = 'water.temperature')
  }, error = function(e){
    cat(paste0(e$message,'\n'), file = results_log, append = TRUE)
    return(10) # a high RMSE value
  })


  # need to compare to obs and return NLL or RMSE stats; using rmse for now
  if (rmse < min_rmse){
    min_rmse <<- rmse
    cat(paste0(min_rmse,',',Sys.time(), '\n'), file = results_log, append = TRUE)
  }


  return(rmse)
}

run_glm_copy <- function(sim_folder, glm_nix_dir, verbose = FALSE){
  origin  <- getwd()
  setwd(sim_folder)
  Sys.setenv(LD_LIBRARY_PATH=file.path(glm_nix_dir, 'nixGLM'))
  glm_path <- file.path(glm_nix_dir, "glm_nix_exe")

  tryCatch({
    if (verbose){
      out <- system2(glm_path, wait = TRUE, stdout = "",
                     stderr = "")
    } else {
      out <- system2(glm_path, wait = TRUE, stdout = NULL,
                     stderr = NULL)
    }
    setwd(origin)
    return(out)
  }, error = function(err) {
    print(paste("GLM_ERROR:  ",err))
    setwd(origin)
  })

}


library(readr)
#using array mode, you have access to the task ID
# which can be used to divide jobs
task_id = as.numeric(Sys.getenv('SLURM_ARRAY_TASK_ID', 'NA'))

if(is.na(task_id)){
  stop("ERROR Can not read task_id, NA returned")
}

job_table <- read_csv('in/fig_3_yeti_jobs.csv')
nml_file <- file.path('in/fig_3', job_table$nml_file[task_id])
driver_file <- file.path('in/fig_3', job_table$meteo_file[task_id])
train_file <- file.path('in/fig_3', job_table$train_file[task_id])
test_file <- file.path('in/fig_3', job_table$test_file[task_id])
exper_id <- job_table$exper_id[task_id]
sim_dir <- file.path(Sys.getenv('LOCAL_SCRATCH', unset="sim-scratch"), sprintf('task_%s_%s', task_id, exper_id))
dir.create(sim_dir, recursive = TRUE)

# it seems to be very slow to run opim on many nodes. Is this because they are all hitting the same GLM exe?
# copy glm exe and use the copy for each sim:
glm_nix_dir <- file.path(sim_dir, 'glm_nix')
dir.create(glm_nix_dir)
file.copy(system.file('exec/nixglm', package='GLMr'), file.path(glm_nix_dir, 'glm_nix_exe'))
file.copy(system.file('extbin/nixGLM', package='GLMr'), glm_nix_dir, recursive = TRUE)

glm_nix_full_dir <- glm_nix_dir
run_optim_glm(driver_file, nml_file, train_file, test_file, exper_id, sim_dir, glm_nix_full_dir)
