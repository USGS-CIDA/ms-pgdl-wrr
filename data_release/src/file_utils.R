bundle_meteo_files <- function(zip_filename, lake_ids, pattern, dir = "../fig_3/yeti_sync"){
  cdir <- getwd()
  on.exit(setwd(cdir))
  files <- get_file_matches(lake_ids, pattern, dir)


  setwd(dir)
  zip(file.path(cdir, zip_filename), files = files)
  setwd(cdir)

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
