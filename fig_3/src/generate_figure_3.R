

plot_calibrated_figure_3 <- function(){

  library(dplyr)
  library(readr)
  library(tidyr)
  jared_pgdl <- readr::read_csv('~/Downloads/WRR Figure3 sparse results - Sheet1 (2).csv')  %>% #'fig_3/in/glm_uncal_vs_PGDL_rmses.csv'
    mutate(PGDL_all = as.numeric(`PGDL - all obs`), DL_all = as.numeric(`DL -all obs`),
           PGDL_10 = as.numeric(`PGDL 10`), DL_10 = as.numeric(`DL 10`),
           PB_uncal = `GLM uncal rmse`,
           nhd_id = paste0('nhd_', nhd_id)) %>% slice(1:68) %>%
    select(nhd_id, PGDL_10, DL_10, PGDL_all, DL_all)


  glm_rmse <- readr::read_tsv('fig_3/in/glm_uncal_rmses.tsv') %>%
    rename(PB_uncal = rmse) %>%
    mutate(nhd_id = paste0('nhd_', as.character(nhd_id))) %>%
    select(nhd_id, PB_uncal)

  plot_data <- left_join(jared_pgdl, glm_rmse, by = "nhd_id")

  optim_files <- dir('yeti_in/fig_3/')
  cal_GLM <- data.frame(nhd_id = c(), PB_cal = c(), exper = c())
  for (file in optim_files){
    this_data <- read_csv(file.path('yeti_in/fig_3/', file)) %>%
      mutate(exper = paste0(strsplit(file, '[_]')[[1]][3], '_', strsplit(file, '[_]')[[1]][6])) %>% select(nhd_id, PB_cal = test_rmse, exper)
    cal_GLM <- rbind(cal_GLM, this_data)
  }

  cal_GLM <- cal_GLM %>% spread(exper, PB_cal) %>%
    rename(PB_10_01 = `010_01`, PB_10_02 = `010_02`, PB_10_03 = `010_03`, PB_all = all_NA) %>%
    rowwise() %>% mutate(PB_10 = median(c(PB_10_01, PB_10_02, PB_10_03), na.rm = TRUE)) %>% select(nhd_id, PB_10, PB_all)
  plot_data <- left_join(plot_data, cal_GLM, by = 'nhd_id') %>% filter(!is.na(PB_all), !is.na(PB_10), !is.na(PGDL_all))


  n_sims <- nrow(plot_data)
  browser()
  png(filename = 'figures/figure_3_wrr.png', width = 9.5, height = 12, units = 'in', res = 200)

  par(omi = c(1,0,0.1,0.1), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.7, 0.5)
  xlim <- c(0.7, 4.55)

  positions <- list(PB_uncal = 1,
                    PB_10 = 1.8,
                    PB_all = 2.2,
                    DL_10 = 2.8,
                    DL_all = 3.2,
                    PGDL_10 = 3.8,
                    PGDL_all = 4.2)

  plot(NA, NA, xlim = xlim, ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)
  library(beanplot)
  bean_w <- 0.45
  plot_data <- plot_data %>% select(-nhd_id)

  beanplot(plot_data$PB_uncal, plot_data$PB_all, plot_data$DL_all, plot_data$PGDL_all, plot_data$PB_10, plot_data$DL_10, plot_data$PGDL_10, maxwidth = bean_w, what=c(0,1,0,0), log = "", add = TRUE,
            axes = F, border = NA, at = c(positions$PB_uncal, positions$PB_all, positions$DL_all, positions$PGDL_all, positions$PB_10, positions$DL_10, positions$PGDL_10))

  med_w <- 0.2
  ind_w <- 0.03
  for (x_bin in 1:ncol(plot_data)){
    mod_name <- names(plot_data)[x_bin]
    segments(x0 = positions[[mod_name]]-ind_w, x1 = positions[[mod_name]]+ind_w, y0 = plot_data[[mod_name]], col = 'black')
    segments(x0 = positions[[mod_name]]-ind_w/2, x1 = positions[[mod_name]]+ind_w/2, y0 = plot_data[[mod_name]], col = 'white')
    segments(x0 = positions[[mod_name]]-med_w/2, x1 = positions[[mod_name]]+med_w/2, y0 = median(plot_data[[mod_name]]), col = 'white', lwd = 2)
    message(round(median(plot_data[[mod_name]]),3), " ", mod_name)
  }


  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)

  axis(1, at = c(-100, 1, 1.8, 2.2, 2.8, 3.2, 3.8, 4.3, 1e10),
       labels = c("", expression("PB"['u']), expression("PB"['10']), expression("PB"['all']),
                  expression("DL"['10']), expression("DL"['all']),
                  expression("PGDL"['10']), expression("PGDL"['all']), ""), tck = -0.01)

  par(mai = c(0.2,0,0,0), mgp = c(2, 2.3,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""), tck = NA, cex.axis = 1.5, lwd = NA)
  par(mai = c(0.2,0,0,0), mgp = c(2,3.8,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), expression("Based"['calibrated']), "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}
