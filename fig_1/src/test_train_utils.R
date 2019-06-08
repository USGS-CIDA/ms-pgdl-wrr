convert_train_move_hash <- function(file_dir, ...){
  feather_files <- c(...)
  data_files <- c()

  for (file in feather_files){
    formatted_filepath <- format_train_filepath(file, file_dir)
    read_feather(file) %>% write_csv(formatted_filepath)
    data_files <- c(data_files, formatted_filepath)
  }
  sc_indicate(ind_file = "", data_file = data_files)
}

convert_test_move_hash <- function(file_dir, ...){
  feather_files <- c(...)
  data_files <- c()

  for (file in feather_files){
    formatted_filepath <- format_test_filepath(file, file_dir)
    read_feather(file) %>% write_csv(formatted_filepath)
    data_files <- c(data_files, formatted_filepath)
  }
  sc_indicate(ind_file = "", data_file = data_files)
}

format_train_filepath <- function(file, file_dir){
  # convert
  # mendota_training_002profiles_experiment_01.feather
  # - to -
  # me_sparse_train_002_profiles_experiment_01.csv

  file %>% basename() %>% tools::file_path_sans_ext() %>%
    str_remove('profiles') %>% str_replace('training', 'train') %>%
    str_replace('mendota','me') %>% str_replace('_experiment', '_profiles_experiment') %>%
    paste0('.csv') %>% file.path(file_dir, .)

}

format_test_filepath <- function(file, file_dir){
  # convert
  # mendota_sparse_test_experiment_01.feather
  # - to -
  # me_sparse_test_experiment_01.csv
  file %>% basename() %>% tools::file_path_sans_ext() %>%
    str_replace('mendota','me') %>%
    paste0('.csv') %>% file.path(file_dir, .)

}

