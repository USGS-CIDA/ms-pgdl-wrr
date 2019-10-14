
plot_data_sparsity <- function(){
  library(dplyr)
  library(readr)
  library(stringr)

  eval_data <- readr::read_csv('data_release/out/me_RMSE.csv', col_types = 'iccd') %>%
    filter(str_detect(exper_id, 'similar_[0-9]+')) %>%
    mutate(col = case_when(
      exper_model == 'pb' ~ '#1b9e77',
      exper_model == 'dl' ~'#d95f02',
      exper_model == 'pgdl' ~ '#7570b3'
    ), pch = case_when(
      exper_model == 'pb' ~ 21,
      exper_model == 'dl' ~ 22,
      exper_model == 'pgdl' ~ 23
    ), n_prof = as.numeric(str_extract(exper_id, '[0-9]+')))

  png(filename = 'figures/figure_1_wrr.png', width = 8, height = 10, units = 'in', res = 200)
  par(omi = c(0,0,0.05,0.05), mai = c(1,1,0,0), las = 1, mgp = c(2,.5,0), cex = 1.5)

  plot(NA, NA, xlim = c(2, 1000), ylim = c(4.7, 0.75),
       ylab = "", xlab = "", log = 'x', axes = FALSE) #'Test RMSE (°C)' "Training temperature profiles (#)"

  n_profs <- c(2, 10, 50, 100, 500, 980)

  axis(1, at = c(-100, n_profs, 1e10), labels = c("", n_profs, ""), tck = -0.01)
  axis(2, at = seq(0,10), las = 1, tck = -0.01)

  # slight horizontal offsets so the markers don't overlap:
  offsets <- data.frame(pgdl = c(0.15, 0.5, 3, 7, 20, 30)) %>%
    mutate(dl = -pgdl, pb = 0, n_prof = n_profs)


  for (mod in c('pb','dl','pgdl')){
    mod_data <- filter(eval_data, exper_model == mod)
    mod_profiles <- unique(mod_data$n_prof)
    for (mod_profile in mod_profiles){
      ._d <- filter(mod_data, n_prof == mod_profile) %>% summarize(y0 = min(rmse), y1 = max(rmse), col = unique(col))
      x_pos <- offsets %>% filter(n_prof == mod_profile) %>% pull(!!mod) + mod_profile
      lines(c(x_pos, x_pos), c(._d$y0, ._d$y1), col = ._d$col, lwd = 2.5)
    }
    ._d <- group_by(mod_data, n_prof) %>% summarize(y = mean(rmse), col = unique(col), pch = unique(pch)) %>%
      rename(x = n_prof) %>% arrange(x)

    lines(._d$x + tail(offsets[[mod]], nrow(._d)), ._d$y, col = ._d$col[1], lty = 'dashed')
    points(._d$x + tail(offsets[[mod]], nrow(._d)), ._d$y, pch = ._d$pch[1], col = ._d$col[1], bg = 'white', lwd = 2.5, cex = 1.5)

  }

  points(2.2, 0.79, col = '#7570b3', pch = 23, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 0.80, 'Process-Guided Deep Learning', pos = 4, cex = 1.1)

  points(2.2, 0.94, col = '#d95f02', pch = 22, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 0.95, 'Deep Learning', pos = 4, cex = 1.1)

  points(2.2, 1.09, col = '#1b9e77', pch = 21, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 1.1, 'Process-Based', pos = 4, cex = 1.1)

  dev.off()

}
