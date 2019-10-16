
jared_pgdl <- readr::read_csv('~/Downloads/Comparing Stopping Conditions - Sheet1.csv')  %>% #'fig_3/in/glm_uncal_vs_PGDL_rmses.csv'
  mutate(PGDL_all = as.numeric(`PGDL(400 ep)`), DL_all = as.numeric(`DL(400 ep)`),
         PGDL_10 = as.numeric(`PGDL_10(400 ep)`), DL_10 = as.numeric(`DL_10(400 ep)`),
         PB_uncal = `GLM uncal rmse`,
         n_train = as.numeric(`train days`),
         nhd_id = paste0('nhd_', nhd_id)) %>% slice(1:68) %>%
  select(nhd_id, PGDL_old = PGDL_all, DL_old = DL_all)

d = read_csv('out/all_RMSE.csv') %>% rename(nhd_id = site_id) %>%
  filter(model_type == 'dl') %>% group_by(nhd_id, exper_id) %>%
  summarize(DL_new = mean(rmse)) %>% left_join(jared_pgdl)

d2 <- read_csv('out/all_RMSE.csv') %>% rename(nhd_id = site_id) %>%
  filter(model_type == 'pgdl') %>% group_by(nhd_id, exper_id) %>%
  summarize(PGDL_new = mean(rmse), PGDL_new_max = max(rmse), PGDL_new_min = min(rmse), PGDL_range = PGDL_new_max-PGDL_new_min) %>%
  left_join(d) %>%
  select(-exper_id) %>% mutate(col = case_when(
    PGDL_old >= PGDL_new_min & PGDL_old <= PGDL_new_max ~ 'black',
    TRUE ~ 'red')
  )

png(filename = '~/Downloads/figure_4b_diagnostic_RNN.png', width =10, height = 24, units = 'in', res = 200)
par(omi = c(0.6,0,0.05,0.05), mai = c(0,1.6,0,0), las = 1, tck = -0.01, cex = 1.5)
plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(3.3, 0.99), ylab = 'Test RMSE (°C)', xlab = '', axes = FALSE)

axis(side = 2, at = seq(0, 10, by = 0.25))
axis(side = 1, at = c(1,2), c('DL_old', 'DL_new'))
box()

segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d2$DL_old, y1 = d2$DL_new, lwd = 1.5)
points(x = rep(1, nrow(d)), y = d2$DL_old, pch = 16, col = 'black', cex = 0.6)
text(x = 1, y = d2$DL_old, labels =d2$nhd_id, cex = 0.8, pos = 2)
points(x = rep(2, nrow(d)), y = d2$DL_new, pch = 16, col = 'black', cex = 0.5)
text(x = 2, y = d2$DL_new, labels = d2$nhd_id, cex = 0.8, pos = 4)
dev.off()

png(filename = '~/Downloads/figure_4b_diagnostic_PGRNN.png', width =10, height = 24, units = 'in', res = 200)
par(omi = c(0.6,0,0.05,0.05), mai = c(0,1.6,0,0), las = 1, tck = -0.01, cex = 1.5)
plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(2.8, 0.97), ylab = 'Test RMSE (°C)', xlab = '', axes = FALSE)

axis(side = 2, at = seq(0, 10, by = 0.25))
axis(side = 1, at = c(1,2), c('PGDL_old', 'PGDL_new'))
box()

segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d2$PGDL_old, y1 = d2$PGDL_new, lwd = 1.5, col = d2$col)
points(x = rep(1, nrow(d2)), y = d2$PGDL_old, pch = 16, col = d2$col, cex = 0.6)
text(x = 1, y = d2$PGDL_old, labels =d2$nhd_id, cex = 0.8, pos = 2, col = d2$col)
points(x = rep(2, nrow(d2)), y = d2$PGDL_new, pch = 16, col = d2$col, cex = 0.5)
text(x = 2, y = d2$PGDL_new, labels = d2$nhd_id, cex = 0.8, pos = 4, col = d2$col)
dev.off()

png(filename = '~/Downloads/figure_4c_diagnostic_PGRNN.png', width =10, height = 10, units = 'in', res = 200)
plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.8), ylim = c(0.7, 2.8), ylab = 'PGDL_new test RMSE (°C)',
     xlab = 'PGDL_new test RMSE (°C)', axes = FALSE)
axis(side = 2, at = seq(0, 10, by = 0.5))
axis(side = 1, at = seq(0, 10, by = 0.5))
box()


abline(0,1)
segments(d2$PGDL_old, x1 = d2$PGDL_old, y0 = d2$PGDL_new_min, y1 = d2$PGDL_new_max, col = d2$col)
points(d2$PGDL_old, d2$PGDL_new, col = d2$col, pch = 21, bg = 'white')
dev.off()

fig_2_old_data <- readr::read_csv('~/Downloads/revision_Figure_1_results - Sheet1.csv') %>%
  rename(old_rmse = `Test RMSE`, exper_n = Experiment, exper_model = Model) %>% select(-`Train RMSE`) %>%
  filter(!exper_model %in% c(NA, "PGRNN")) %>% mutate(exper_model = case_when(
    exper_model == 'GLM' ~ "pb",
    exper_model == 'RNN' ~ 'dl',
    exper_model == 'PGRNN_pretrained_prev_yrs' ~ 'pgdl',
    TRUE ~ NA_character_
  ))


fig_2 = read_csv('data_release/out/me_RMSE.csv') %>% rowwise() %>% mutate(n_profiles = {as.numeric(strsplit(exper_id, '[_]')[[1]][2])}) %>%
  ungroup() %>% filter(substr(exper_id, start = 1, stop = 7) == 'similar') %>% select(-exper_id) %>% rename(new_rmse = rmse, exper_model = model_type) %>%
  left_join(fig_2_old_data, by = c('exper_model','exper_n','n_profiles')) %>%
  mutate(col = case_when(
    n_profiles == 980 ~ '#e41a1c',
    n_profiles == 500 ~ '#377eb8',
    n_profiles == 100 ~ '#4daf4a',
    n_profiles == 50 ~ '#984ea3',
    n_profiles == 10 ~ '#ff7f00',
    n_profiles == 2 ~ '#e7298a'
  ))


png(filename = '~/Downloads/figure_2_diagnostic_PGRNN.png', width =10, height = 8, units = 'in', res = 200)
par(omi = c(0.2,0.2,0.05,0.05), mai = c(0.2,0.35,0,0), las = 1, mgp = c(2.2,0.8,0), tck = -0.03, xpd=TRUE)
layout(mat = matrix(c(1,2,3,4,5,6), nrow = 1))


for (p in c(980, 500, 100, 50, 10, 2)){

  plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(2.2, 0.89), ylab = '', xlab = '', axes = FALSE)
  if (p == 980){
    text(0.4, 1.5, srt = 90, 'Test RMSE (°C)', cex = 2)
    axis(side = 2, at = c(1, 1.25, 1.75, 2))
    par(xpd=FALSE)
  } else {
    axis(side = 2, at = c(1, 1.25, 1.5, 1.75, 2))
  }
  axis(side = 1, at = c(1,2), c('PGDL_old', 'PGDL_new'))
  box()

  d <- filter(fig_2, exper_model == 'pgdl', n_profiles == p)
  segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d$old_rmse, y1 = d$new_rmse, lwd = 1.5, col = d$col)
  points(x = rep(1, nrow(d)), y = d$old_rmse, pch = 16, col = 'black', cex = 0.6)
  text(x = 1, y = d$old_rmse, labels = paste('e#', d$exper_n, sep=''), cex = 0.8, pos = 2)
  points(x = rep(2, nrow(d)), y = d$new_rmse, pch = 16, col = 'black', cex = 0.5)
  text(x = 2, y = d$new_rmse, labels = paste('e#', d$exper_n, sep=''), cex = 0.8, pos = 4)
  text(1.5, 0.87, sprintf("%s profiles", p), cex = 2)
}

dev.off()


png(filename = '~/Downloads/figure_2_diagnostic_RNN.png', width =9, height = 8, units = 'in', res = 200)
par(omi = c(0.2,0.2,0.05,0.05), mai = c(0.2,0.35,0,0), las = 1, mgp = c(2.2,0.8,0), tck = -0.03, xpd=TRUE)
layout(mat = matrix(c(1,2,3,4,5), nrow = 1))


for (p in c(980, 500, 100, 50, 10)){

  plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(3.9, 0.89), ylab = '', xlab = '', axes = FALSE)
  if (p == 980){
    text(0.4, 2.5, srt = 90, 'Test RMSE (°C)', cex = 2)
    axis(side = 2, at = c(1, 1.25,1.5, 1.75, 2, 3, 3.25, 3.5, 3.75, 4))
    par(xpd=FALSE)
  } else {
    axis(side = 2, at = seq(1, 5, by = 0.25))
  }
  axis(side = 1, at = c(1,2), c('DL_old', 'DL_new'))
  box()

  d <- filter(fig_2, exper_model == 'dl', n_profiles == p)
  segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d$old_rmse, y1 = d$new_rmse, lwd = 1.5, col = d$col)
  points(x = rep(1, nrow(d)), y = d$old_rmse, pch = 16, col = 'black', cex = 0.6)
  text(x = 1, y = d$old_rmse, labels = paste('e#', d$exper_n, sep=''), cex = 0.8, pos = 2)
  points(x = rep(2, nrow(d)), y = d$new_rmse, pch = 16, col = 'black', cex = 0.5)
  text(x = 2, y = d$new_rmse, labels = paste('e#', d$exper_n, sep=''), cex = 0.8, pos = 4)
  text(1.5, 0.87, sprintf("%s profiles", p), cex = 2)
}

dev.off()





fig_3_old_data <- readr::read_csv('~/Downloads/revision_Figure_2_results - Sheet1.csv') %>%
  rename(old_rmse = `Test RMSE`, exper_n = Experiment, exper_model = Model) %>% select(-`Train RMSE`) %>%
  filter(!exper_model %in% c(NA, "PGRNN")) %>% mutate(exper_model = case_when(
    exper_model == 'GLM' ~ "pb",
    exper_model == 'RNN' ~ 'dl',
    exper_model == 'PGRNN_pretrained_prev_yrs' ~ 'pgdl',
    TRUE ~ NA_character_
  )) %>% rowwise() %>% mutate(exper_type = strsplit(exper_n, '[_]')[[1]][1], lake_name = strsplit(exper_n, '[_]')[[1]][3], lake_id = case_when(
    lake_name == 'sparkling' ~ 'sp',
    lake_name == 'mendota' ~ 'me',
    TRUE ~ NA_character_
  ), exper_type = case_when(
    exper_type == 'seasons' ~ 'season',
    exper_type == 'years' ~ 'year',
    exper_type == 'similar' ~ 'similar'
  ), exper_n = {as.numeric(strsplit(exper_n, '[_]')[[1]][2])}) %>% ungroup() %>% select(exper_n, exper_model, old_rmse, exper_type, lake_id)

fig_3 <- purrr::map(c('data_release/out/sp_RMSE.csv', 'data_release/out/me_RMSE.csv'), function(x) {
  lake_id <- strsplit(basename(x), '[_]')[[1]][1]
  read_csv(x) %>% filter(!exper_id %in% c('similar_2', 'similar_10', 'similar_50', 'similar_100', 'similar_980')) %>% rowwise() %>% mutate(exper_type = {strsplit(exper_id, '[_]')[[1]][1]}) %>%
    ungroup() %>% select(-exper_id) %>% rename(new_rmse = rmse, exper_model = model_type) %>% mutate(lake_id = lake_id)
  }) %>% purrr::reduce(rbind) %>%
  left_join(fig_3_old_data, by = c('exper_n','exper_model','exper_type','lake_id')) %>%
  mutate(col = case_when(
    exper_type == 'season' ~ '#e41a1c',
    exper_type == 'year' ~ '#377eb8',
    exper_type == 'similar' ~ '#4daf4a'
  ))



png(filename = '~/Downloads/figure_3_diagnostic_PGRNN.png', width =7.5, height = 8, units = 'in', res = 200)
par(omi = c(0.2,0.2,0.05,0.05), mai = c(0.2,0.35,0,0), las = 1, mgp = c(2.2,0.8,0), tck = -0.03, xpd=TRUE)
layout(mat = matrix(c(1,2,3), nrow = 1))


for (type in c('similar','year','season')){

  plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(3.25, 0.89), ylab = '', xlab = '', axes = FALSE)
  if (type == 'similar'){
    text(0.45, 2.5, srt = 90, 'Test RMSE (°C)', cex = 1.8)
    par(xpd=FALSE)
    axis(side = 2, at = c(1, 1.25,1.5, 1.75, 2, 3, 3.25, 3.5, 3.75, 4))
  } else {
    axis(side = 2, at = seq(1, 5, by = 0.25))
  }
  axis(side = 1, at = c(1,2), c('PGDL_old', 'PGDL_new'))
  box()

  d <- filter(fig_3, exper_model == 'pgdl', exper_type == !!type)
  segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d$old_rmse, y1 = d$new_rmse, lwd = 1.5, col = d$col)
  points(x = rep(1, nrow(d)), y = d$old_rmse, pch = 16, col = 'black', cex = 0.6)
  text(x = 1, y = d$old_rmse, labels = paste(d$lake_id, '_e#', d$exper_n, sep=''), cex = 0.8, pos = 2)
  points(x = rep(2, nrow(d)), y = d$new_rmse, pch = 16, col = 'black', cex = 0.5)
  text(x = 2, y = d$new_rmse, labels = paste(d$lake_id, '_e#', d$exper_n, sep=''), cex = 0.8, pos = 4)
  text(1.5, 0.87, type, cex = 2)
}

dev.off()



png(filename = '~/Downloads/figure_3_diagnostic_RNN.png', width =7.5, height = 8, units = 'in', res = 200)
par(omi = c(0.2,0.2,0.05,0.05), mai = c(0.2,0.35,0,0), las = 1, mgp = c(2.2,0.8,0), tck = -0.03, xpd=TRUE)
layout(mat = matrix(c(1,2,3), nrow = 1))


for (type in c('similar','year','season')){

  plot(c(NA,0), c(NA,NA), xlim = c(0.7, 2.3), ylim = c(3.25, 0.89), ylab = '', xlab = '', axes = FALSE)
  if (type == 'similar'){
    text(0.45, 2.5, srt = 90, 'Test RMSE (°C)', cex = 1.8)
    par(xpd=FALSE)
    axis(side = 2, at = c(1, 1.25,1.5, 1.75, 2, 3, 3.25, 3.5, 3.75, 4))
  } else {
    axis(side = 2, at = seq(1, 5, by = 0.25))
  }
  axis(side = 1, at = c(1,2), c('DL_old', 'DL_new'))
  box()

  d <- filter(fig_3, exper_model == 'dl', exper_type == !!type)
  segments(x0 = rep(1,68), x1 = rep(2, 68), y0 = d$old_rmse, y1 = d$new_rmse, lwd = 1.5, col = d$col)
  points(x = rep(1, nrow(d)), y = d$old_rmse, pch = 16, col = 'black', cex = 0.6)
  text(x = 1, y = d$old_rmse, labels = paste(d$lake_id, '_e#', d$exper_n, sep=''), cex = 0.8, pos = 2)
  points(x = rep(2, nrow(d)), y = d$new_rmse, pch = 16, col = 'black', cex = 0.5)
  text(x = 2, y = d$new_rmse, labels = paste(d$lake_id, '_e#', d$exper_n, sep=''), cex = 0.8, pos = 4)
  text(1.5, 0.87, type, cex = 2)
}

dev.off()


files <- data.frame(files = dir('../fig_3/in'), stringsAsFactors = FALSE) %>%
  filter(str_detect(files, 'nhd_[0-9]+_temperatures.feather')) %>% pull(files)
cdir <- getwd()
setwd('../fig_3/in')
zip(zipfile = "~/Desktop/share/glm_calibrated_outputs.zip", files)
setwd(cdir)


#re-write these to zip
lakes <- remake::make("modeled_lakes") %>% sf::st_as_sf() %>% sf::st_transform(2811) %>%
  mutate(perim = lwgeom::st_perimeter_2d(geometry), area = sf::st_area(geometry), circle_perim = 2*pi*sqrt(area/pi),
         SDF = perim/circle_perim) %>% sf::st_drop_geometry() %>% rowwise() %>% mutate(canopy_height = mda.lakes::getCanopy(site_id)) %>% ungroup() %>% select(site_id, SDF, canopy_height) %>%
  feather::write_feather('~/Desktop/share/jared_metadata_10_10.feather')
