bundle_meteo_files <- function(zip_filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  cdir <- getwd()
  on.exit(setwd(cdir))
  files <- get_file_matches(lake_ids, pattern, dir)


  setwd(dir)
  zip(file.path(cdir, zip_filename), files = files)
  setwd(cdir)

}

load_npy_df <- function(filepath){
  npyLoad(filepath) %>% t() %>% as.data.frame() %>%
    mutate(date = seq(as.Date("2009-04-02"), length.out = 3185, by = 'days')) %>%
    select(date, everything())
}

combine_XJ_npy_similar <- function(fileout, n_depths, ...){
  npy_files <- c(...)

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
    exp_n <- file_splits[4] %>% str_remove('[^0-9.]+')

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

exper_id_from_file <- function(filename){
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

sb_replace_files <- function(sb_id, ...){
  item_replace_files(sb_id, files = c(...))
}

merge_obs_files <- function(filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  files <- get_file_matches(lake_ids, pattern, dir)
  out <- data.frame(site_id = c(), DateTime = c(), Depth = c(), temp = c(), stringsAsFactors = FALSE)


  for (file in files){
    data <- read_csv(file.path(dir, file), col_types = 'Ddd') %>%
      mutate(site_id = site_id_from_file(file)) %>% select(site_id, everything())
    out <- rbind(out, data)
  }

  write_csv(out, path = filename)

}

merge_single_lake_obs_files <- function(filename, pattern, dir, exper_id){

  files <- data.frame(filename = dir(dir), stringsAsFactors = FALSE) %>%
    filter(stringr::str_detect(string = filename, pattern = pattern)) %>%
    pull(filename)

  out <- data.frame(DateTime = c(), Depth = c(), temp = c(), exper_n = c(), exper_id = c(), stringsAsFactors = FALSE)

  for (file in files){

    data <- read_csv(file.path(dir, file), col_types = 'Ddd') %>%
      mutate(exper_n = exper_n_from_file(file),
             exper_id = exper_id_from_file(file),
             prof_n = prof_n_from_file(file)) %>%
      mutate(exper_id = paste0(exper_id, "_", prof_n)) %>% select(-prof_n)
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
