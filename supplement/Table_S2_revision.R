library(feather)
library(dplyr)
library(googledrive)
# read in table s1 to get nhdid to lakeid crosswalk
metadata_table <- read.csv('D:/R Projects/lake-temp-supplement/lake_metadata_table.csv', stringsAsFactors = FALSE)

# get nhdids of 68 lakes
# files downloaded from https://drive.google.com/drive/u/1/folders/1uf2SMfQ5NbemV22FvrfNYulTvnqrLSbN
files <- list.files("D:/R Projects/lake-temp-supplement/updated_model_runs_20190117")
temp_files <- grep(pattern = 'test_train', files, value = TRUE)

# get data all together
# this is the data that were used for experiment 3
used_dat <- data.frame()
for (i in 1:length(temp_files)) {
  temp_dat <- read_feather(file.path('D:/R Projects/lake-temp-supplement/updated_model_runs_20190117', temp_files[i]))
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
drive_download(as_id('https://docs.google.com/spreadsheets/d/1CiYLoV8169xeZuHNtzuLfm-vAoMpTALgfCL97PrVvdE/edit#gid=0'),
               path = 'D:/R Projects/lake-temp-supplement/fig3_rmse_data.csv', overwrite = T)

rmse_dat <- read.csv('D:/R Projects/lake-temp-supplement/fig3_rmse_data.csv', stringsAsFactors = F,nrows = 68)
rmse_dat$nhd_id <- paste0('nhd_', rmse_dat$nhd_id)

rmse_dat <- mutate(rmse_dat,
                   `GLM (pre-trainer)`= round(`GLM.uncal.rmse`,2),
                   `GLM (calibrated)` = round(`GLM.cal`,2),
                   PGDL = round(PGDL, 2), `PGDL (50)` = round(PGDL_50,2), `PGDL (10)` = round(PGDL_10,2),
                   DL = round(DL, 2), `DL (50)` = round(DL_50,2), `DL (10)` = round(DL_10,2)) %>%
  select(nhd_id, `GLM (pre-trainer)`, `GLM (calibrated)`,PGDL, `PGDL (50)`, `PGDL (10)`,DL, `DL (50)`, `DL (10)`)


# merge in RMSE data
summary68 <- left_join(summary68,
                       rmse_dat)

# write.csv(summary68, 'supplement/out/temp_obs_summary_update.csv', row.names = F)
#  file <- drive_upload(
#    'supplement/out/temp_obs_summary_update.csv',
#    path = as_id('https://drive.google.com/drive/u/1/folders/1yCCcqfPeppdQM79adK5dWrUmMzIwBvQv'),
#    name = "Table_S2.csv",
#    type = "spreadsheet"
#  )

drive_update(file = as_id('https://docs.google.com/spreadsheets/d/1msBsOu92fqT3NuEdfb-OHVMpxULb879qXEEZpaV6qFU/edit#gid=1537497571'),
             media = 'D:/R Projects/lake-temp-supplement/temp_obs_summary_update.csv.csv')

