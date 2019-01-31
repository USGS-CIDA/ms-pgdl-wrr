library(googledrive)
library(lakeattributes)

################################
# make table of lake characteristics (Table S1)
################################

# get lake nhdid to lake names crosswalk
lake_metadata <- readRDS('D:/R Projects/lake-temp-supplement/feature_crosswalk.rds')

# get nhdids of 68 lakes
# files downloaded from https://drive.google.com/drive/u/1/folders/1uf2SMfQ5NbemV22FvrfNYulTvnqrLSbN
files <- list.files("D:/R Projects/lake-temp-supplement/updated_model_runs_20190117")
temp_files <- grep(pattern = 'test_train', files, value = TRUE)
lakes <- unique(gsub('_test_train.feather', '', x = temp_files))

# Table S1 variables
area <- as.numeric(sapply(lakes, get_area))
latlong <- sapply(lakes, get_latlon)
maxdepth <- as.numeric(sapply(lakes, get_zmax))
lake_metadata$site_id <- as.character(lake_metadata$site_id)
names.db <- lake_metadata[, c('site_id', 'GNIS_Nm')]
names.db <- unique(names.db)

names <- left_join(data.frame(site_id = lakes), names.db)
names <- left_join(names, missing_names) %>%
  mutate(name = ifelse(is.na(GNIS_Nm), lake_name, as.character(GNIS_Nm)))

kd <- get_kd_avg(lakes, default.if.null = FALSE)

bathy_hyps <- lapply(lakes, get_bathy, cone_est = FALSE)
cone_estimated <- sapply(bathy_hyps, is.null)
bathy <- lapply(lakes, get_bathy, cone_est = FALSE)

meandepth <- function(bathy) {

  diff_depth <- diff(bathy$depths)
  diff_depth <- diff_depth/3
  area0 <- as.numeric(bathy$areas[-nrow(bathy)])
  area1 <- as.numeric(bathy$areas[-1])
  area_int <- area0+area1+sqrt(area0*area1)
  volume <- sum(diff_depth*area_int)
  zmean <- volume/bathy$areas[1]

  return(zmean)

}

zmean <- c()

for (i in 1:length(bathy)) {
  if (is.null(bathy[[i]])){
    zmean[i] <- NA
  } else {
    zmean[i] <- meandepth(bathy[[i]])
  }
}

metadata_table <- data.frame(nhd_id = names$site_id,
                             lake_name = as.character(names$name),
                             latitude = as.numeric(latlong[1, ]),
                             longitude = as.numeric(latlong[2, ]),
                             maxdepth = round(maxdepth,1),
                             meandepth = round(zmean, 1),
                             area_km2 = round(area/1000000, 2),
                             kd_avg = round(kd$kd_avg, 2), stringsAsFactors = FALSE)

# check for duplicate lake names
dupes <- group_by(metadata_table, lake_name) %>%
  summarize(count = n())

# clean up lake names
metadata_table$lake_name <- gsub('\\s[[:digit:]]+', '', metadata_table$lake_name)

# need to rename duplicate names, will do this by latitude
fish_names <- c('Fish Lake (1)', 'Fish Lake (2)', 'Fish Lake (3)')
cedar_names <- c('Cedar Lake (1)', 'Cedar Lake (2)')
silver_names <- c('Silver Lake (1)', 'Silver Lake (2)')

metadata_table$lake_name[metadata_table$lake_name %in% 'Fish Lake'] <- fish_names
metadata_table$lake_name[metadata_table$lake_name %in% 'Cedar Lake'] <- cedar_names
metadata_table$lake_name[metadata_table$lake_name %in% 'Silver Lake'] <- silver_names

write.csv(metadata_table, 'D:/R Projects/lake-temp-supplement/lake_metadata_table.csv', row.names = F)

# original write to google drive

#file <- drive_upload(
#  '~/Downloads/lake_metadata_table.csv',
#  name = "lake_metadata_upload.csv",
#  type = "spreadsheet"
#)

# update if already written
drive_update(file = as_id('https://docs.google.com/spreadsheets/d/1FPFi4QSnlIZkutrEQlapYhX5mkhEwiQrtQq3zFiPo3c/edit#gid=1256307542'), 'D:/R Projects/lake-temp-supplement/lake_metadata_table.csv')
