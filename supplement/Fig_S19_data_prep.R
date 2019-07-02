# this function depends on Table S2
# rerun Table_S2_revision if you need to update
format_figS19_dat <- function() {


  library(dplyr)
  library(tidyr)
  library(googledrive)

  dir.create('supplement/in', showWarnings = FALSE)

  # get RMSE data from Table S2
  drive_download(as_id('https://docs.google.com/spreadsheets/d/1lSTsUqcC5CSzTRyXTFi1K8Yaabd9yhW24OVONUpu_34'),
                 path = 'supplement/in/table_s2.csv', overwrite = T)

  summary68 <- read.csv('supplement/in/table_s2.csv', check.names = FALSE, nrows = 68)

  # get lake names from Table S1
  drive_download(as_id('https://docs.google.com/spreadsheets/d/1FPFi4QSnlIZkutrEQlapYhX5mkhEwiQrtQq3zFiPo3c/edit#gid=88060065'), path = 'supplement/in/lake_metadata_table.csv', overwrite = T)

  metadata_table <- read.csv('supplement/in/lake_metadata_table.csv', stringsAsFactors = F,nrows = 68)

  rmse_order <- summary68 %>%
    arrange(`PGDL(400 ep)`) %>%
    mutate(nhd_id = paste0('nhd_', nhd_id)) %>%
    left_join(select(metadata_table, nhd_id, lake_name))

  rmse_long <- select(summary68,-num_dates, -`train days`, -`test days`, -`#Train Obs`, -`#Test Obs`) %>%
    gather(key = model, value = RMSE, -nhd_id) %>%
    filter(model %in% c('DL_10(400 ep)', 'DL(400 ep)', 'PGDL(400 ep)',
                        'PGDL_10(400 ep)', 'GLM uncal rmse', 'GLM cal')) %>%
    mutate(nhd_id = paste0('nhd_', nhd_id)) %>%
    left_join(select(metadata_table, nhd_id, lake_name)) %>%
    mutate(model_10 = ifelse(grepl('_10', model), TRUE, FALSE))

  rmse_long$lake_name <- factor(rmse_long$lake_name, levels = rmse_order$lake_name)

  return(rmse_long)
}

