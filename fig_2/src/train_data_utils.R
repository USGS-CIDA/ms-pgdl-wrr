

build_sparse_training <- function(filepath, buoy_data, chunksize, remove_chunks, exp_n, prof_n){


  if (!'data.frame' %in% class(buoy_data)){
    buoy_data <- feather::read_feather(buoy_data)
  }

  un_dates <- buoy_data %>% pull(DateTime) %>% unique

  un_dt_resample <- data.frame(date = un_dates, remove = FALSE)

  set.seed(42 + as.numeric(exp_n))

  for (i in seq_len(remove_chunks)){
    good_sample <- FALSE
    while (!good_sample){
      start_i <- sample(1:nrow(un_dt_resample), 1)
      end_i <- start_i + chunksize - 1
      if (end_i <= nrow(un_dt_resample) & !any(un_dt_resample$remove[start_i:end_i])){
        good_sample = TRUE
        un_dt_resample$remove[start_i:end_i] <- TRUE
      }
    }
  }

  buoy_data %>% filter(DateTime %in% sample(un_dt_resample$date[!un_dt_resample$remove], size = prof_n, replace = FALSE)) %>%
    arrange(DateTime) %>% group_by(DateTime, Depth) %>% summarize(temp = mean(temp)) %>%
    write_csv(path = filepath)
}

training_hash_files <- function(exper_name, buoy_data, chunksize, remove_chunks, mask_filepath = NULL){


  details <- strsplit(exper_name, '[_]')[[1]]
  n_exp <- as.numeric(details[5])
  prof_n <- as.numeric(details[3])

  # file patterns like: me_sparse_train_500_profiles_experiment_01.csv
  data_files <- c()
  for (exp in 1:n_exp){
    this_buoy_data <- buoy_data
    if (!is.null(mask_filepath)){
      stopifnot(all(details[1:2] == c('sp','similar')))
      these_masks <- feather::read_feather(mask_filepath) %>% filter(experiment_detail == sprintf('sparkling_2a_experiment_%s', stringr::str_pad(exp, width = 2, pad = '0')))
      for (j in seq_len(nrow(these_masks))){
        mask_start <- these_masks$mask_start[j]
        mask_end <- these_masks$mask_end[j]
        this_buoy_data <- this_buoy_data %>% filter(DateTime < mask_start | DateTime > mask_end, !is.na(temp))
      }
    }
    this_filepath <- paste(c(details[1:2], 'train', details[3:4], 'experiment', stringr::str_pad(exp, width = 2, pad = '0')), collapse = '_') %>%
      paste0('.csv') %>% file.path('fig_2/yeti_sync', .)
    build_sparse_training(this_filepath, buoy_data = this_buoy_data, chunksize, remove_chunks, exp_n = exp, prof_n = prof_n)
    data_files <- c(data_files, this_filepath)
  }

  sc_indicate(ind_file = "", data_file = data_files)
}

test_hash_files <- function(exper_name, buoy_data, mask_filepath = NULL){
  details <- strsplit(exper_name, '[_]')[[1]]
  n_exp <- as.numeric(details[4])
  this_buoy_data <- buoy_data[0L, ]

  mask_names <- c("sp" = 'sparkling_2a_experiment_%s',
                  "me" = 'mendota_2a_experiment_%s')

  data_files <- c()

  for (exp in 1:n_exp){
    this_buoy_data <- buoy_data[0L, ]
    if (!is.null(mask_filepath)){
      stopifnot(details[2] == 'similar')
      these_masks <- feather::read_feather(mask_filepath) %>% filter(experiment_detail == sprintf(mask_names[[details[1]]], stringr::str_pad(exp, width = 2, pad = '0')))
      for (j in seq_len(nrow(these_masks))){
        mask_start <- these_masks$mask_start[j]
        mask_end <- these_masks$mask_end[j]
        this_buoy_data <- filter(buoy_data, DateTime >= mask_start & DateTime <= mask_end, !is.na(temp)) %>% rbind(this_buoy_data) %>% arrange(DateTime)
      }
    } else {
      this_buoy_data <- buoy_data
    }
    this_buoy_data <- this_buoy_data %>% arrange(DateTime) %>% group_by(DateTime, Depth) %>% summarize(temp = mean(temp))

    this_filepath <- paste(c(details[1:3], 'experiment', stringr::str_pad(exp, width = 2, pad = '0')), collapse = '_') %>%
      paste0('.csv') %>% file.path('fig_2/yeti_sync', .)
    write_csv(this_buoy_data, path = this_filepath)
    data_files <- c(data_files, this_filepath)
  }

  sc_indicate(ind_file = "", data_file = data_files)

}

collapse_dot_names <- function(...){
  names(c(...))
}
