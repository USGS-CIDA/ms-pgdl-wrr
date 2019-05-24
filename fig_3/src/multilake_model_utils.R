
get_write_driver <- function(driverpath, nhd_id){
  # locate the driver, copy to location
  localpath <- readRDS("~/Downloads/feature_nldas_coords.rds") %>% filter(site_id == nhd_id) %>%
    mutate(driver_file = sprintf('../lake-temperature-model-prep/7_drivers_munge/out/NLDAS_time[0.351500]_x[%s]_y[%s].csv', nldas_coord_x, nldas_coord_y)) %>%
    pull(driver_file)
  drivers <- driver_add_rain(read_csv(localpath), rain_add = 0.7) %>%
    readr::write_csv(path = driverpath)
}

export_geometry <- function(filepath, site_id){
  stopifnot(packageVersion('lakeattributes') == '0.10.2')
  get_bathy(site_id) %>% write_csv(path = filepath)
}

run_export_glm <- function(filepath, glm_nml, ...){
  nml_sim_path <- 'glm2.nml'
  write_nml(glm_nml = glm_nml, file = nml_sim_path)

  lake_depth <- get_nml_value(glm_nml, arg_name = 'lake_depth')
  export_depths <- seq(0, lake_depth, by = 0.5)
  run_glm('.', verbose = FALSE)

  temp_data <- get_temp(reference = 'surface', z_out = export_depths)
  model_out <- get_var(var_name = 'hice') %>%
    mutate(ice = hice > 0) %>% select(-hice) %>%
    left_join(temp_data, ., by = 'DateTime')
  feather::write_feather(model_out, filepath)
  unlink(x = c('lake.csv','glm2.nml','overflow.csv','output.nc'))
}

get_set_base_nml <- function(nhd_id, driverpath, ...){
  stopifnot(packageVersion('lakeattributes') == '0.10.2')
  nml <- populate_base_lake_nml(site_id = nhd_id, driver = driverpath) %>%
    set_nml(arg_list = list(...))
  nml$sed_heat <- NULL

  return(nml)

}
