library(dplyr)
library(ggplot2)
library(tidyr)
library(googledrive)

##############################
# Experiment 3 supp fig
##############################
# get RMSE data from Table S2
summary68 <- read.csv('D:/R Projects/lake-temp-supplement/temp_obs_summary.csv',check.names = FALSE)

rmse_order <- mutate(summary68, rmse_diff = `GLM (calibrated)` - `PGDL`) %>%
  arrange(rmse_diff)

rmse_long <- gather(summary68, key = model, value = RMSE, `GLM (pre-trainer)`, `PGDL`, `GLM (calibrated)`)

rmse_long$lake_name <- factor(rmse_long$lake_name, levels = rmse_order$lake_name)

p <- ggplot(rmse_long, aes(x = lake_name, y = RMSE)) +
  geom_point(aes(color = model, shape = model), size = 1.7, fill = 'white') +
  scale_color_manual(values = c('#1b9e77','#1b9e77', '#7570b3')) + # match Jordan's colors in fig 3
  scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
  theme_bw() +
  coord_cartesian(ylim = c(0.7, 5.5)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        legend.position = c(0.086,0.15),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        legend.title = element_blank(),
        legend.background = element_rect(colour = 'black',
                                         fill = 'white', linetype='solid'),
        legend.text = element_text(size = 8),
        legend.key.height = unit(0.5, 'line')) +
  labs(x = '') +
  scale_y_reverse()

ggsave('D:/R Projects/lake-temp-supplement/WRR_sup_S19_all_lakes_PB_PGDL.png', p,
       height = 100, width = 230, units = 'mm', scale = 0.9)

#drive_upload(media = 'D:/R Projects/lake-temp-supplement/WRR_sup_S19_all_lakes_PB_PGDL.png',
#             path = as_id('https://drive.google.com/drive/u/1/folders/1E2c9VnqEW6oKyDctb1O-hnUKODpXo6Jh/'))

drive_update(file = as_id('https://drive.google.com/drive/u/1/folders/1E2c9VnqEW6oKyDctb1O-hnUKODpXo6Jh'),
             media = 'D:/R Projects/lake-temp-supplement/WRR_sup_S17_all_lakes_PB_PGDL.png')
