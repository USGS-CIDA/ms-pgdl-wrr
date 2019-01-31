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
drive_download(as_id('https://docs.google.com/spreadsheets/d/18NGzzxsBBYebs4A091UTvnpDu-zB3ZSu62CzbHE-vEg/edit#gid=1597158640'),
               path = 'D:/R Projects/lake-temp-supplement/fig3_rmse_data.csv', overwrite = T)

rmse_dat <- read.csv('D:/R Projects/lake-temp-supplement/fig3_rmse_data.csv', stringsAsFactors = F,nrows = 68)
rmse_dat$nhd_id <- paste0('nhd_', rmse_dat$nhd_id)

# get RMSE data for calibrated GLM
drive_download(as_id('https://docs.google.com/spreadsheets/d/1hJLUujM5KE8ZKswSPPwol9DNpr0NIyoTDE5RitpGqvI/edit#gid=0'),
               path = 'D:/R Projects/lake-temp-supplement/fig3_rmse_glm_data.csv', overwrite = T)

rmse_glm <- read.csv(file = 'D:/R Projects/lake-temp-supplement/fig3_rmse_glm_data.csv',
                     stringsAsFactors = F)
names(rmse_glm) <- c('nhd_id', 'glm_uncal_rmse', 'glm_cal_rmse')

# get RMSE data for uncalibrated GLM
# data sent by Alison via email
rmse_uncal_glm <- read.table('D:/R Projects/lake-temp-supplement/glm_uncal_rmses.tsv', header = T)
rmse_uncal_glm$nhd_id <- paste0('nhd_', rmse_uncal_glm$nhd_id)
names(rmse_uncal_glm)[5] <- 'glm_uncal_rmse'


# merge in RMSE data
summary68 <- left_join(summary68,
                       select(rmse_dat, nhd_id, Test.Begin, Test.End, Train.Begin, Train.End, PGDL.rmse)) %>%
  left_join(select(rmse_glm, nhd_id, glm_cal_rmse)) %>%
  left_join(select(rmse_uncal_glm, nhd_id, glm_uncal_rmse)) %>%
  mutate(`Test Period` = paste0(`Test.Begin`, ' - ', `Test.End`),
         `Train Period` = paste0(`Train.Begin`, ' - ', `Train.End`)) %>%
  mutate(`GLM (pre-trainer)` = round(glm_uncal_rmse, 2),
         `GLM (calibrated)` = round(glm_cal_rmse, 2),
         `PGDL` = round(`PGDL.rmse`, 2)) %>%
  select(-`Train.Begin`, -`Train.End`, -`Test.Begin`, -`Test.End`, -glm_uncal_rmse, -`PGDL.rmse`, -nhd_id, -glm_cal_rmse)

write.csv(summary68, 'D:/R Projects/lake-temp-supplement/temp_obs_summary.csv', row.names = F)
# file <- drive_upload(
#   '~/Downloads/temp_obs_summary.csv',
#   name = "temp_obs_summary_upload.csv",
#   type = "spreadsheet"
# )

drive_update(file = as_id('https://docs.google.com/spreadsheets/d/11NuiDjpgsiQ02CivnhopxH0s7AsPUcKpW_X4nc2Iry0/edit#gid=1217264599'),
             media = 'D:/R Projects/lake-temp-supplement/temp_obs_summary.csv')

