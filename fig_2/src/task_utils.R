
create_fig2_job_table <- function(target_name, ...){
  #site_id,nml_file,meteo_file,exper_id,train_file,test_file

  #job_table <- data.frame(site_id = c(), nml_file = c(), meteo_file = c(), exper_id = c(), train_file = c(), test_file = c())
  files <- c(...) %>% basename()
  train_files <- files[stringr::str_detect(files, "train")]

  get_split_id <- function(train_file, split_indx){
    sapply(train_file, FUN = function(x) strsplit(x, '[_]')[[1]][split_indx], USE.NAMES = FALSE)
  }

  get_x_file <- function(site_id, pattern){
    x_files <- str_subset(files, pattern)
    files_out <- site_id
    for (i in 1:length(files_out)){
      files_out[i] <- str_subset(x_files, sprintf('%s%s', site_id[i], pattern))
    }
    return(files_out)
  }

  get_exper_id <- function(site_id, train_file){
    # like sp_season_500_profiles_experiment_02
    exper_id <- site_id
    for (i in 1:length(site_id)){
      details <- strsplit(train_file[i], '[_]')[[1]] %>% .[-3] # remove "train"
      details[length(details)] <- tail(details, 1) %>% tools::file_path_sans_ext()
      exper_id[i] <- paste(details, collapse = '_')
    }
    return(exper_id)
  }


  data.frame(train_file = train_files, stringsAsFactors = FALSE) %>%
    mutate(site_id = get_split_id(train_file, 1),
           exper_id = get_exper_id(site_id, train_file),
           nml_file = get_x_file(site_id, '_nml.'),
           meteo_file = get_x_file(site_id, '_meteo.'),
           challenge_id = get_split_id(train_file, 2),
           exp_n = get_split_id(train_file, 7),
           test_file = sprintf("%s_%s_test_experiment_%s", challenge_id, site_id, exp_n)) %>%
    select(site_id, nml_file, meteo_file, exper_id, train_file, test_file) %>%
    readr::write_csv(target_name)
}
