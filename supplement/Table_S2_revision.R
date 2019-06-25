
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
  files <- list.files("D:/R Projects/lake-temp-supplement/updated_model_runs_20190117")
  temp_files <- grep(pattern = 'test_train', files, value = TRUE)

    # this is the data that were used for experiment 3
  used_dat <- data.frame()
  for (i in 1:length(temp_files)) {
    temp_dat <- feather::read_feather(file.path('D:/R Projects/lake-temp-supplement/updated_model_runs_20190117', temp_files[i]))
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

  drive_download(as_id('https://docs.google.com/spreadsheets/d/1CiYLoV8169xeZuHNtzuLfm-vAoMpTALgfCL97PrVvdE/edit#gid=0'),
                 path = 'supplement/in/fig3_rmse_data.csv', overwrite = T)

  rmse_dat <- read.csv('supplement/in/fig3_rmse_data.csv', stringsAsFactors = F,nrows = 68)
  rmse_dat$nhd_id <- paste0('nhd_', rmse_dat$nhd_id)

  # currently do not have the 10/50 experiments in here
  rmse_dat <- mutate(rmse_dat,
                     `GLM (pre-trainer)`= round(`GLM.uncal.rmse`,2),
                     `GLM (calibrated)` = round(`GLM.cal`,2),
                     PGDL = round(PGDL, 2),
                     #`PGDL (50)` = round(PGDL_50,2), `PGDL (10)` = round(PGDL_10,2),
                     DL = round(DL, 2),
                     #`DL (50)` = round(DL_50,2), `DL (10)` = round(DL_10,2)
  ) %>%
    select(nhd_id, `GLM (pre-trainer)`,
           `GLM (calibrated)`,
           DL,
           #`DL (10)`,
           PGDL,
           #`PGDL (10)`,
    )


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

  drive_update(file = as_id('https://docs.google.com/spreadsheets/d/1msBsOu92fqT3NuEdfb-OHVMpxULb879qXEEZpaV6qFU/edit#gid=1537497571'),
               media = 'supplement/out/temp_obs_summary_update.csv')
}
