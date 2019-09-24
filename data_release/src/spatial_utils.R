

subset_save_winslow <- function(lake_ids){

  sb_id <- '57d97341e4b090824ffb0e6f'
  layer <- 'model_lakes'
  url <- 'https://www.sciencebase.gov/catalogMaps/mapping/ows/%s?service=wfs&request=GetFeature&typeName=sb:%s&outputFormat=shape-zip&version=1.0.0'
  destination = tempfile(pattern = 'lake_shape', fileext='.zip')
  query <- sprintf(url, sb_id, layer)
  file <- GET(query, write_disk(destination, overwrite=T), progress())
  shp.path <- tempdir()
  unzip(destination, exdir = shp.path)

  out <- readOGR(shp.path, layer=layer) %>%
    spTransform(CRSobj = CRS("+init=epsg:4326 +proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

  out <- out[out$site_id %in% lake_ids$site_id, 1]
  return(out)
}
