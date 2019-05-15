

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
  cal_GLM <- readr::read_tsv('fig_3/in/GLM cal figure 3 - Sheet1.tsv') %>% rename(nhd_id = `NHD id`, calibrated = `test RMSE`)
  cal_GLM$nhd_id = sapply(1:nrow(cal_GLM), FUN = function(i) strsplit(cal_GLM$nhd_id[i], '[_]')[[1]][2])

  plot_data <- left_join(plot_data, cal_GLM, by = 'nhd_id') %>%
    filter(!is.na(calibrated)) %>% select(nhd_id, GLM, PGDL, calibrated)

  n_sims <- nrow(plot_data)

  png(filename = 'figures/figure_3_wrr.png', width = 9.5, height = 12, units = 'in', res = 200)

  par(omi = c(0.5,0,0.1,0.8), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.1, 0.65)

  plot(NA, NA, xlim = c(0.9, 4.1), ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)

  # TEMPORARY UNTIL MODELS ARE RUN!!
  plot_data$rnn <- plot_data$calibrated + rnorm(n = n_sims, sd = 0.1, mean = -0.2)

  diffs <- data.frame(cal = plot_data$GLM - plot_data$calibrated,
                      ml = plot_data$calibrated - plot_data$rnn,
                      pg = plot_data$rnn - plot_data$PGDL)

  min_dif <- sapply(names(diffs), function(x) min(diffs[[x]]), USE.NAMES = FALSE) %>% min %>% round(1)
  max_dif <- sapply(names(diffs), function(x) max(diffs[[x]]), USE.NAMES = FALSE) %>% max %>% round(1)


  neg_bins <- seq(min_dif, to = 0, by = 0.1)
  pos_bins <- seq(0.1, to = max_dif, by = 0.1)
  col_df <- data.frame(bin = as.character(c(neg_bins, pos_bins)),
                       col = c(colorRampPalette(c('#d73027','#fc8d59','#fee08b','#e5e5ab'))(length(neg_bins)), tail(colorRampPalette(c('#e5e5ab','#d9ef8b','#91cf60','#1a9850'))(length(pos_bins)+1), -1L))) %>%
    mutate(col = paste0(col, ''))



  seg <- data.frame(bin = as.character(round(diffs$cal, 1))) %>% left_join(col_df)
  segments(x0 = rep(1, n_sims), x1 = rep(2, n_sims), y0 = plot_data$GLM, y1 = plot_data$calibrated, col = seg$col,
           lwd = 3)#approx(x = c(0, max(abs(min_dif), max_dif)), y = c(0.2, 6), xout = abs(diffs$cal))$y)
  seg <- data.frame(bin = as.character(round(diffs$ml, 1))) %>% left_join(col_df)
  segments(x0 = rep(2, n_sims), x1 = rep(3, n_sims), y0 = plot_data$calibrated, y1 = plot_data$rnn, col = seg$col,
           lwd = 3)#approx(x = c(0, max(abs(min_dif), max_dif)), y = c(0.1, 6), xout = abs(diffs$ml))$y)
  seg <- data.frame(bin = as.character(round(diffs$pg, 1))) %>% left_join(col_df)
  segments(x0 = rep(3, n_sims), x1 = rep(4, n_sims), y0 = plot_data$rnn, y1 = plot_data$PGDL, col = seg$col,
           lwd = 3)#approx(x = c(0, max(abs(min_dif), max_dif)), y = c(0.1, 6), xout = abs(diffs$pg))$y)


  points(rep(1, n_sims), plot_data$GLM, col = 'black', pch = 16, cex = 0.8, lwd = 1.5)
  points(rep(2, n_sims), plot_data$calibrated, col = 'black', pch = 16, cex = 0.8, lwd = 1.5)
  points(rep(2, n_sims), plot_data$calibrated, col = 'black', pch = 16, cex = 0.8, lwd = 1.5)
  points(rep(3, n_sims), plot_data$rnn, col = 'black', pch = 16, cex = 0.8, lwd = 1.5)
  points(rep(4, n_sims), plot_data$PGDL, col = 'black', pch = 16, cex = 0.8, lwd = 1.5)




  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2, 0.8,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""), tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0.2,0,0,0), mgp = c(2,2.3,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), expression("Based"['calibrated']), "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}
