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

  glm_rmse <- readr::read_tsv('fig_3/in/glm_uncal_rmses.tsv') %>%
    rename(PB_uncal = rmse) %>%
    mutate(nhd_id = paste0('nhd_', as.character(nhd_id))) %>%
    select(nhd_id, PB_uncal)

  summary68 <- left_join(summary68, glm_rmse) %>% select(-`GLM (pre-trainer)`) %>% select(`GLM (pre-trainer)` = PB_uncal, everything())

  rmse_order <- mutate(summary68, rmse_diff = `GLM (calibrated)` - `PGDL`) %>%
    arrange(rmse_diff)

  rmse_long <- select(summary68,-`Unique observations`, -Days, -Years) %>%
    gather(key = model, value = RMSE, -lake_name, -nhd_id) %>%
    mutate(model_10 = ifelse(grepl('_10', model), TRUE, FALSE))

  rmse_long$lake_name <- factor(rmse_long$lake_name, levels = rmse_order$lake_name)

  return(rmse_long)
}

