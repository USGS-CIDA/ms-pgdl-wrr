
generate_tableS2 <- function(){

  library(feather)
  library(dplyr)
  library(googledrive)

  # read in table s1 to get nhdid to lake name crosswalk
  dir.create('supplement/in', showWarnings = FALSE)
  drive_download(as_id('https://docs.google.com/spreadsheets/d/1FPFi4QSnlIZkutrEQlapYhX5mkhEwiQrtQq3zFiPo3c/edit#gid=88060065'),
                 path = 'supplement/in/lake_metadata_table.csv', overwrite = T)
  metadata_table <- read.csv('supplement/in/lake_metadata_table.csv', stringsAsFactors = F,nrows = 68)


  # get model run data
  # files downloaded from https://drive.google.com/drive/u/1/folders/1uf2SMfQ5NbemV22FvrfNYulTvnqrLSbN
  files <- list.files("C:/Users/soliver/Documents/R Projects/lake-temp-supplement/updated_model_runs_20190117")
  temp_files <- grep(pattern = 'test_train', files, value = TRUE)

    # this is the data that were used for experiment 3
  used_dat <- data.frame()
  for (i in 1:length(temp_files)) {
    temp_dat <- feather::read_feather(file.path('C:/Users/soliver/Documents/R Projects/lake-temp-supplement/updated_model_runs_20190117', temp_files[i]))
    used_dat <- bind_rows(used_dat, temp_dat)

  }

  # summarize experiment 3 data
  total_obs <- group_by(used_dat, nhd_id) %>%
    summarize(total_obs = n())

  total_lakedays <- group_by(used_dat, nhd_id, date) %>%
    summarize(ndepths = n()) %>%
    group_by(nhd_id) %>%
    summarize(ndays = n())

  nperyear <- used_dat %>%
    group_by(nhd_id, date) %>%
    summarize(ndepths = n()) %>%
    mutate(year = lubridate::year(date)) %>%
    group_by(nhd_id, year) %>%
    summarize(nperyear = n())

  year_summary <- nperyear %>%
    group_by(nhd_id) %>%
    summarize(nyears = n(), avgnperyear = mean(nperyear))

  # bring together
  summary68 <- total_obs %>%
    left_join(total_lakedays) %>%
    left_join(year_summary) %>%
    left_join(select(metadata_table, nhd_id, lake_name)) %>%
    select(nhd_id, lake_name, total_obs, ndays, nyears) %>%
    rename('Unique observations' = total_obs,
           'Days' = ndays,
           "Years" = nyears) %>%
    arrange(-`Unique observations`)

  # get PGDL RMSE data
  dir.create('supplement/in', showWarnings = FALSE)

  drive_download(as_id('https://docs.google.com/spreadsheets/d/1lSTsUqcC5CSzTRyXTFi1K8Yaabd9yhW24OVONUpu_34/edit#gid=0'),
                 path = 'supplement/in/fig3_rmse_data.csv', overwrite = T)

  rmse_dat <- read.csv('supplement/in/fig3_rmse_data.csv', stringsAsFactors = F,nrows = 68, check.names = FALSE)
  rmse_dat$nhd_id <- paste0('nhd_', rmse_dat$nhd_id)

  # get GLM data
  glm_dat <- read.csv('supplement/in/calibrated_GLM_wrr_revision.csv')

  # merge rmse dat
  all_rmse <- left_join(rmse_dat, glm_dat)

  rmse_dat <- mutate(all_rmse,
                     PB_0 = round(`GLM uncal rmse`,2),
                     PB_10 = round(PB_10, 2),
                     PB_all = round(PB_all,2),
                     PGDL_all = round(`PGDL(400 ep)`, 2),
                     PGDL_10 = round(`PGDL_10(400 ep)`, 2),
                     DL_all = round(`DL(400 ep)`, 2),
                     DL_10 = round(`DL_10(400 ep)`, 2)) %>%
    select(nhd_id,PB_0, PB_10, PB_all, DL_10, DL_all, PGDL_10, PGDL_all)


  # merge in RMSE data
  summary68 <- left_join(summary68,
                         rmse_dat)

  dir.create('supplement/out', showWarnings = FALSE)

  write.csv(summary68, 'supplement/out/temp_obs_summary_update.csv', row.names = F)
  #  file <- drive_upload(
  #    'supplement/out/temp_obs_summary_update.csv',
  #    path = as_id('https://drive.google.com/drive/u/1/folders/1yCCcqfPeppdQM79adK5dWrUmMzIwBvQv'),
  #    name = "Table_S2.csv",
  #    type = "spreadsheet"
  #  )

  drive_update(file = as_id('1msBsOu92fqT3NuEdfb-OHVMpxULb879qXEEZpaV6qFU'),
               media = 'supplement/out/temp_obs_summary_update.csv')
}
