# this function depends on Table S2
# rerun Table_S2_revision if you need to update
format_figS19_dat <- function() {


  library(dplyr)
  library(tidyr)
  library(googledrive)

  dir.create('supplement/in', showWarnings = FALSE)

  # get RMSE data from Table S2
  drive_download(as_id('https://docs.google.com/spreadsheets/d/1msBsOu92fqT3NuEdfb-OHVMpxULb879qXEEZpaV6qFU/edit#gid=1424852496'),
                 path = 'supplement/in/table_s2.csv', overwrite = T)

  summary68 <- read.csv('supplement/in/table_s2.csv', check.names = FALSE)

  rmse_order <- mutate(summary68, rmse_diff = `GLM (calibrated)` - `PGDL`) %>%
    arrange(rmse_diff)

  rmse_long <- select(summary68,-`Unique observations`, -Days, -Years) %>%
    gather( key = model, value = RMSE, -nhd_id, -lake_name) %>%
    mutate(model_type = case_when(
      grepl('GLM', model) ~ 'GLM',
      grepl('PGDL', model) ~ 'PGDL',
      TRUE ~ 'DL'
    ))

  rmse_long$lake_name <- factor(rmse_long$lake_name, levels = rmse_order$lake_name)

  return(rmse_long)
}

