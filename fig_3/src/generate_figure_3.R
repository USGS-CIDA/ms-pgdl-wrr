

plot_calibrated_figure_3 <- function(){

  library(dplyr)

  jared_pgdl <- readr::read_csv('~/Downloads/glm_uncal_vs_PGDL_rmses - glm_uncal_rmses.csv')  %>% #'fig_3/in/glm_uncal_vs_PGDL_rmses.csv'
    mutate(PGDL = as.numeric(`PGDL - partial PT`), rnn = `DL - no PT`) %>% slice(1:68)


  glm_rmse <- readr::read_tsv('fig_3/in/glm_uncal_rmses.tsv') %>%
    rename(GLM = rmse) %>%
    mutate(nhd_id = as.character(nhd_id)) %>%
    select(nhd_id, GLM)

  plot_data <- left_join(jared_pgdl, glm_rmse, by = "nhd_id")

  # from https://docs.google.com/spreadsheets/d/1hJLUujM5KE8ZKswSPPwol9DNpr0NIyoTDE5RitpGqvI/edit#gid=0
  cal_GLM <- readr::read_tsv('fig_3/in/GLM cal figure 3 - Sheet1.tsv') %>% rename(nhd_id = `NHD id`, calibrated = `test RMSE`)
  cal_GLM$nhd_id = sapply(1:nrow(cal_GLM), FUN = function(i) strsplit(cal_GLM$nhd_id[i], '[_]')[[1]][2])

  plot_data <- left_join(plot_data, cal_GLM, by = 'nhd_id') %>%
    filter(!is.na(calibrated), !is.na(rnn)) %>% select(nhd_id, GLM, PGDL, rnn, calibrated)

  n_sims <- nrow(plot_data)

  png(filename = 'figures/figure_3_wrr.png', width = 9.5, height = 12, units = 'in', res = 200)

  par(omi = c(0.5,0,0.1,0.1), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.6, 0)
  xlim <- c(0.5, 4.5)

  plot(NA, NA, xlim = xlim, ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)
  library(beanplot)

  plot_data <- select(plot_data, GLM, calibrated, rnn, PGDL)

  beanplot(plot_data$GLM, plot_data$calibrated, plot_data$rnn, plot_data$PGDL, maxwidth = 0.95, what=c(0,1,0,0), log = "", add = TRUE,
           axes = F, border = NA)#col = list(c('#1b9e7799','#7570b399'), c('#7570b399', '#1b9e7799')), lwd = 3,

  med_w <- 0.3
  ind_w <- 0.04
  for (x_bin in 1:ncol(plot_data)){
    mod_name <- names(plot_data)[x_bin]
    segments(x0 = x_bin-ind_w/2, x1 = x_bin+ind_w/2, y0 = plot_data[[mod_name]], col = 'white')
    segments(x0 = x_bin-med_w/2, x1 = x_bin+med_w/2, y0 = median(plot_data[[mod_name]]), col = 'white', lwd = 2)
  }


  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2, 0.8,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""), tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2,2.3,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), expression("Based"['calibrated']), "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}
