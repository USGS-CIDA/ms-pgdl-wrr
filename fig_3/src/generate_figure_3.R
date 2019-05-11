

plot_calibrated_figure_3 <- function(){

  library(dplyr)

  jared_pgdl <- readr::read_csv('fig_3/in/glm_uncal_vs_PGDL_rmses.csv')  %>%
    mutate(PGDL = as.numeric(`PGDL rmse`)) %>% slice(1:68)


  glm_rmse <- readr::read_tsv('fig_3/in/glm_uncal_rmses.tsv') %>%
    rename(GLM = rmse) %>%
    mutate(nhd_id = as.character(nhd_id)) %>%
    select(nhd_id, GLM)

  plot_data <- left_join(jared_pgdl, glm_rmse, by = "nhd_id")

  # from https://docs.google.com/spreadsheets/d/1hJLUujM5KE8ZKswSPPwol9DNpr0NIyoTDE5RitpGqvI/edit#gid=0
  cal_GLM <- readr::read_tsv('fig_3/in/GLM cal figure 3 - Sheet1.tsv') %>% rename(nhd_id = `NHD id`, calibrated = `test RMSE`) %>%
    mutate(nhd_id = sapply(1:nrow(cal_GLM), FUN = function(i) strsplit(cal_GLM$nhd_id[i], '[_]')[[1]][2]))

  plot_data <- left_join(plot_data, cal_GLM, by = 'nhd_id') %>%
    filter(!is.na(calibrated)) %>% select(nhd_id, GLM, PGDL, calibrated)

  n_sims <- nrow(plot_data)

  png(filename = 'figures/figure_3_wrr.png', width = 9.5, height = 12, units = 'in', res = 200)

  par(omi = c(0.5,0,0.1,0.8), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.1, 0.65)

  plot(NA, NA, xlim = c(0.9, 4.1), ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)

  # TEMPORARY UNTIL MODELS ARE RUN!!
  plot_data$rnn <- sample(plot_data$calibrated, size = n_sims, replace = FALSE) - 0.1

  segments(x0 = rep(1, n_sims), x1 = rep(2, n_sims), y0 = plot_data$GLM, y1 = plot_data$calibrated, col = '#00000033', lwd = 3)
  segments(x0 = rep(2, n_sims), x1 = rep(3, n_sims), y0 = plot_data$calibrated, y1 = plot_data$rnn, col = '#00000033', lwd = 3)
  segments(x0 = rep(3, n_sims), x1 = rep(4, n_sims), y0 = plot_data$rnn, y1 = plot_data$PGDL, col = '#00000033', lwd = 3)


  points(rep(1, n_sims), plot_data$GLM, col = '#1b9e77', pch = 8, cex = 0.8, lwd = 1.5)
  points(rep(2, n_sims), plot_data$calibrated, bg = 'white', col = '#1b9e77', pch = 21, cex = 1.5, lwd = 2)
  points(rep(2, n_sims), plot_data$calibrated, bg = 'NA', col = '#1b9e77', pch = 21, cex = 1.5, lwd = 2)
  points(rep(3, n_sims), plot_data$rnn, bg = 'white', col = '#d95f02', pch = 22, cex = 1.5, lwd = 2)
  points(rep(4, n_sims), plot_data$PGDL, bg = 'white', col = '#7570b3', pch = 23, cex = 1.5, lwd = 2)




  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2, 0.8,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""), tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2,2.3,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), "Based", "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}
