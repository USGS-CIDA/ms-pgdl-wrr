bundle_meteo_files <- function(zip_filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  cdir <- getwd()
  on.exit(setwd(cdir))
  files <- get_file_matches(lake_ids, pattern, dir)
  for (file in files){
    read_csv(file.path(dir, file), col_types = 'Dddddddd') %>%
      rename(date = time) %>% write_csv(file.path(tempdir(), file))
  }


  setwd(tempdir())
  zip(file.path(cdir, zip_filename), files = files)
  setwd(cdir)

}

rewrite_meteo <- function(filein, fileout){
  read_csv(filein, col_types = 'Dddddddd') %>%
    rename(date = time) %>% write_csv(fileout)
}

combine_jared_feathers <- function(fileout, ...){

  feather_files <- c(...)
  file_names <- basename(feather_files)

  dir_path <- dirname(feather_files) %>% unique()

  test_exp <- 'historical'
  n_prof <- 'all'

  data <- purrr::map(file_names, function(x) {
    file_splits <- strsplit(x, '[_]')[[1]]

    # get 1 from "trial0" (since we're 1 indexed)
    exp_n <- file_splits[3] %>% strsplit('[.]') %>% .[[1]] %>% .[1] %>%
      str_remove('[^0-9.]+') %>% as.numeric() %>% {.+1}
    # get "nhd_10596466" from "10596466PGRNN"
    lake_id <- file_splits[1] %>% str_remove('[^0-9.]+') %>% paste0('nhd_', .)


    data <- feather::read_feather(file.path(dir_path, x)) %>%
      mutate(date = as.Date(lubridate::ceiling_date(date, 'days')))
    n_depths <- ncol(data) - 1
    # rename and adjust depths
    data %>% setNames(c("date", paste0('temp_',seq(0, length.out = n_depths, by = 0.5)))) %>%
      filter(!is.na(temp_0)) %>% # files start w/ NAs??
      mutate(exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof))
  }) %>% reduce(rbind)

  write_csv(data, path = fileout)
}

glm_feather_to_csv <- function(fileout, ...){
  feather_file <- c(...)
  test_exp <- 'historical'
  n_prof <- 'all'
  exp_n <- 1
  feather::read_feather(feather_file) %>%
    select(-ice, date = DateTime) %>%
    mutate(date = as.Date(lubridate::ceiling_date(date, 'days')),
           exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof)) %>%
    write_csv(path = fileout)
}

convert_glm_to_csv <- function(fileout, min_date, filepath){
  feather::read_feather(filepath) %>%
    select(-ice, date = DateTime) %>%
    mutate(date = as.Date(lubridate::ceiling_date(date, 'days'))) %>%
    filter(date > min_date) %>%
    write_csv(path = fileout)
}

zip_ice <- function(zip_filename, ...){

  cdir <- getwd()
  on.exit(setwd(cdir))
  files <- c(...)


  setwd(unique(dirname(files)))
  zip(file.path(cdir, zip_filename), files = basename(files))
  setwd(cdir)
}

ice_from_GLM_feather <- function(fileout, min_date, filepath){
  feather::read_feather(filepath) %>%
    select(date = DateTime, ice) %>%
    mutate(date = as.Date(lubridate::ceiling_date(date, 'days'))) %>%
    filter(date > min_date) %>%
    write_csv(path = fileout)
}

combine_glm_feather_other <- function(fileout, min_date, ...){
  feather_files <- c(...)
  n_prof = 500

  data <- purrr::map(feather_files, function(x) {
    file_splits <- basename(x) %>% strsplit('[_]') %>% .[[1]]
    # get "season" from "me_season_500_profiles_experiment_01_temperatures.feather":
    test_exp <- file_splits[2]
    # get 1 from "PGRNN_mendota_season_exp1.npy":
    exp_n <- tail(file_splits, 2) %>% head(1) %>% as.numeric()

    feather::read_feather(x) %>%
      select(-ice, date = DateTime) %>%
      mutate(date = as.Date(lubridate::ceiling_date(date, 'days')), exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof)) %>%
      filter(date > min_date)
  }) %>% reduce(rbind)

  write_csv(data, path = fileout)
}

combine_glm_feather_similar <- function(fileout, min_date, ...){
  feather_files <- c(...)
  test_exp <- "similar"

  data <- purrr::map(feather_files, function(x) {
    file_splits <- basename(x) %>% strsplit('[_]') %>% .[[1]]
    # get 2 from "me_002_profiles_experiment_01_temperatures.feather":
    n_prof <- file_splits[2] %>% as.numeric()
    # get 1 from "me_002_profiles_experiment_01_temperatures.feather":
    exp_n <- file_splits[5] %>% as.numeric()

    feather::read_feather(x) %>%
      select(-ice, date = DateTime) %>%
      mutate(date = as.Date(date), exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof)) %>%
      filter(date > min_date)
  }) %>% reduce(rbind)

  write_csv(data, path = fileout)

}

load_npy_df <- function(filepath){
  npyLoad(filepath) %>% t() %>% as.data.frame() %>%
    mutate(date = seq(as.Date("2009-04-02"), length.out = 3185, by = 'days')) %>%
    select(date, everything())
}

combine_XJ_npy_similar <- function(fileout, n_depths, ...){
  npy_files <- file.path(getwd(), c(...))

  dir_path <- dirname(npy_files) %>% unique()
  file_names <- basename(npy_files)
  test_exp <- 'similar'
  data <- purrr::map(file_names, function(x) {
    # get 2 from "02.npy":
    n_prof <- strsplit(x, '[_]') %>% .[[1]] %>% .[4] %>%
      strsplit('[.]') %>% .[[1]] %>% .[1] %>% as.numeric()
    # get 1 from "exp1":
    exp_n <- strsplit(x, '[_]') %>% .[[1]] %>% .[3] %>% str_remove('[^0-9.]+')

    load_npy_df(file.path(dir_path, x)) %>%
      setNames(c("date", paste0('temp_',seq(0, length.out = n_depths, by = 0.5)))) %>%
      mutate(exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof))
    }) %>% reduce(rbind)

  write_csv(data, path = fileout)
}


combine_XJ_npy_other <- function(fileout, n_depths, ...){
  npy_files <- c(...)

  dir_path <- dirname(npy_files) %>% unique()
  file_names <- basename(npy_files)
  n_prof = 500
  data <- purrr::map(file_names, function(x) {
    file_splits <- strsplit(x, '[_]') %>% .[[1]]
    # get "season" from "PGRNN_mendota_season_exp1.npy" & "PGRNN_season_sparkling_exp2.npy":
    if ('mendota' %in% file_splits){
      test_exp <- file_splits[3]
    } else {
      test_exp <- file_splits[2]
    }

    # get 1 from "PGRNN_mendota_season_exp1.npy":
    exp_n <- file_splits[4] %>% strsplit('[.]') %>% .[[1]] %>% .[1] %>%  str_remove('[^0-9.]+')

    load_npy_df(file.path(dir_path, x)) %>%
      setNames(c("date", paste0('temp_',seq(0, length.out = n_depths, by = 0.5)))) %>%
      mutate(exper_n = exp_n, exper_id = sprintf("%s_%s", test_exp, n_prof))
  }) %>% reduce(rbind)

  write_csv(data, path = fileout)
}

get_file_matches <- function(lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  files <- data.frame(filename = dir(dir), stringsAsFactors = FALSE) %>%
    filter(stringr::str_detect(string = filename, pattern = pattern)) %>%
    pull(filename)

  if (files %>% str_match(lake_ids$site_id) %>% is.na() %>% any()){
    stop('something is wrong. not all ids matched files')
  }
  return(files)
}

site_id_from_file <- function(filename){
  filename <- basename(filename)
  paste(strsplit(filename,'[_]')[[1]][1:2], collapse = '_')
}

exper_n_from_file <- function(filename){
  filename <- basename(filename)
  tail(strsplit(filename,'[_]')[[1]],1) %>% strsplit('[.]') %>% .[[1]] %>% .[1] %>% as.numeric
}

exper_type_from_file <- function(filename){
  filename <- basename(filename)
  splits <- strsplit(filename,'[_]')[[1]]
  if (length(splits) == 6 & all(splits[1:2] == c('me','train'))){
    'similar'
  } else {
    splits[2]
  }
}

prof_n_from_file <- function(filename){
  filename <- basename(filename)
  splits <- strsplit(filename,'[_]')[[1]]
  if (length(splits) == 6){
    as.numeric(splits[3])
  } else {
    as.numeric(splits[4])
  }

}

sb_replace_files <- function(sb_id, ..., file_hash){

  if (!sbtools::is_logged_in()){
    sb_secret <- dssecrets::get_dssecret("cidamanager-sb-srvc-acct")
    sbtools::authenticate_sb(username = sb_secret$username, password = sb_secret$password)
  }

  hashed_filenames <- c()
  if (!missing(file_hash)){
    hashed_filenames <- yaml.load_file(file_hash) %>% names
    for (file in hashed_filenames){
      item_replace_files(sb_id, files = file)
    }
  }
  files <- c(...)
  if (length(files) > 0){
    item_replace_files(sb_id, files = files)
  }

}

ice_from_diagnostic <- function(fileout, diag_feather){
  read_feather(diag_feather) %>% mutate(date = as.Date(lubridate::ceiling_date(time, 'days'))) %>%
    mutate(ice = Vol.Black.Ice > 0) %>% select(date, ice) %>%
    write_csv(path = fileout)
}

zip_grouped_hashed_files <- function(hash_filename, hashed_files, group_by, suffix_with){

  # since the `zip()` function seems to only work by getting into the dir of the files to be zipped, we'll reset dir on exit:
  cdir <- getwd()
  on.exit(setwd(cdir))

  file_info <- data.frame(hashed_filenames = names(yaml.load_file(hashed_files)), stringsAsFactors = FALSE) %>%
    mutate(group_name = str_extract(hashed_filenames, group_by), # we'll assume the zipped files go in the same dir as the hash_filename:
           out_filename = file.path(dirname(hash_filename), sprintf('%s%s.zip', group_name, suffix_with)))

  # for each output zip file, group and compress the corresponding files, regardless of what dirs they are in
  zipped_files_out <- purrr::map(unique(file_info$out_filename), function(x){

    filepaths_to_zip <- filter(file_info, out_filename == x) %>% pull(hashed_filenames)
    dir_out <- dirname(filepaths_to_zip) %>% unique()

    # will break if we are trying to combine files that are in different directories. Should work if we mix dirs as long as there aren't differences from within the groups:
    stopifnot(length(dir_out) == 1)
    setwd(file.path(cdir, dir_out))
    zip(file.path(cdir, x), files = basename(filepaths_to_zip))
    setwd(cdir)
    return(x)
  }) %>% purrr::reduce(c) %>%
    sc_indicate(ind_file = hash_filename, data_file = .) # write a hash file that points to each zip file. We'll use this later when uploading each zip file.
}


merge_obs_files <- function(filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  files <- get_file_matches(lake_ids, pattern, dir)
  out <- data.frame(site_id = c(), date = c(), depth = c(), temp = c(), stringsAsFactors = FALSE)


  for (file in files){
    data <- read_csv(file.path(dir, file), col_types = 'Ddd') %>%
      rename(date = DateTime, depth = Depth) %>%
      mutate(site_id = site_id_from_file(file)) %>% select(site_id, everything())
    out <- rbind(out, data)
  }

  write_csv(out, path = filename)

}


merge_multi_lake_test_files <- function(filename, pattern, dir){
  files <- data.frame(filename = dir(dir), stringsAsFactors = FALSE) %>%
    filter(stringr::str_detect(string = filename, pattern = pattern)) %>%
    pull(filename)

  # need lake name, "all" (for n_profiles) or equivalent
  data <- purrr::map(files, function(x) {

    "nhd_1099136_test_all_profiles.csv"
    file_splits <- strsplit(x, '[_]')[[1]]
    # get "nhd_1099136"
    lake_id <- paste0(file_splits[1], '_', file_splits[2])

    read_csv(file.path(dir, x), col_types = 'Ddd') %>%
      rename(date = DateTime, depth = Depth) %>%
      mutate(site_id = lake_id, exper_type = "historical") # add exper_n and exper_type
  }) %>% purrr::reduce(rbind)

  write_csv(data, path = filename)
}

merge_single_lake_test_files <- function(filename, pattern, dir){
  files <- data.frame(filename = dir(dir), stringsAsFactors = FALSE) %>%
    filter(stringr::str_detect(string = filename, pattern = pattern)) %>%
    pull(filename)


  data <- purrr::map(files, function(x) {
    # "sp_similar_test_experiment_03.csv":
    file_splits <- strsplit(x, '[_]')[[1]]
    # get "similar" from "sp_similar_test_experiment_03.csv"
    test_exp <- file_splits[2]
    # get 5 from "05.csv":
    exp_n <- tail(file_splits, 1) %>% strsplit('[.]') %>% .[[1]] %>% .[1] %>% as.numeric()

    read_csv(file.path(dir, x), col_types = 'Ddd') %>%
      rename(date = DateTime, depth = Depth) %>%
      mutate(exper_n = exp_n, exper_type = test_exp) # add exper_n and exper_type
  }) %>% purrr::reduce(rbind)

  write_csv(data, path = filename)
}

merge_single_lake_obs_files <- function(filename, pattern, dir, exper_id){

  files <- data.frame(filename = dir(dir), stringsAsFactors = FALSE) %>%
    filter(stringr::str_detect(string = filename, pattern = pattern)) %>%
    pull(filename)

  out <- data.frame(date = c(), depth = c(), temp = c(), exper_n = c(), exper_id = c(), stringsAsFactors = FALSE)

  for (file in files){

    data <- read_csv(file.path(dir, file), col_types = 'Ddd') %>%
      rename(date = DateTime, depth = Depth) %>%
      mutate(exper_n = exper_n_from_file(file),
             exper_type = exper_type_from_file(file),
             prof_n = prof_n_from_file(file)) %>%
      mutate(exper_id = paste0(exper_type, "_", prof_n)) %>% select(-prof_n, -exper_type)
    out <- rbind(out, data)
  }
  write_csv(out, path = filename)

}

bundle_nml_files <- function(json_filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  files <- get_file_matches(lake_ids, pattern, dir)

  out_list <- vector("list", length = nrow(lake_ids)) %>% setNames(lake_ids$site_id)

  for (id in names(out_list)){
    this_nml_file <- files[str_detect(files, id)] %>% file.path(dir, .)
    nml <- read_nml(nml_file = this_nml_file) %>% unclass()
    out_list[[id]] <- nml
  }

  RJSONIO::toJSON(out_list, pretty = TRUE) %>% write(json_filename)
}


sp_to_zip <- function(zip_filename, sp_object){
  cdir <- getwd()
  on.exit(setwd(cdir))
  dsn <- tempdir()
  layer <- 'pgdl_lakes'
  rgdal::writeOGR(sp_object, dsn = dsn, layer = layer, driver="ESRI Shapefile", overwrite_layer = TRUE)

  files_to_zip <- data.frame(filepath = dir(dsn, full.names = TRUE), stringsAsFactors = FALSE) %>%
    mutate(filename = basename(filepath)) %>%
    filter(str_detect(string = filename, pattern = layer)) %>% pull(filename)

  setwd(dsn)
  zip(file.path(cdir, zip_filename), files = files_to_zip)
  setwd(cdir)
}
