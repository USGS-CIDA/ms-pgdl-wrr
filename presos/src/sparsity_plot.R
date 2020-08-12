get_sb_csv_data <- function(sb_id, file_name){

  temp_file <- tempfile(tools::file_path_sans_ext(file_name), fileext = paste0('.', tools::file_ext(file_name)))
  item_file_download(sb_id = sb_id,
                     names = file_name,
                     destinations = temp_file)

  readr::read_csv(temp_file, col_types = 'iccd')
}


plot_sparsity_WRR <- function(fileout, rmse_fl, pb_col, dl_col, pgdl_col, width = 8, height = 10, res = 250 ){
  eval_data <- readr::read_csv(rmse_fl, col_types = 'iccd') %>%
    filter(str_detect(exper_id, 'similar_[0-9]+')) %>%
    mutate(col = case_when(
      model_type == 'pb' ~ pb_col,
      model_type == 'dl' ~ dl_col,
      model_type == 'pgdl' ~ pgdl_col
    ), pch = case_when(
      model_type == 'pb' ~ 21,
      model_type == 'dl' ~ 22,
      model_type == 'pgdl' ~ 23
    ), n_prof = as.numeric(str_extract(exper_id, '[0-9]+')))

  png(file = fileout, width = width, height = height, units = 'in', res = res)
  par(omi = c(0,0,0.05,0.05), mai = c(1,1,0,0), las = 1, mgp = c(2,.5,0), cex = 1.5)

  plot(NA, NA, xlim = c(2, 1000), ylim = c(4.7, 0.75),
       ylab = "Test RMSE (°C)", xlab = "Training temperature profiles (#)", log = 'x', axes = FALSE) #'Test RMSE (°C)' "Training temperature profiles (#)"

  n_profs <- c(2, 10, 50, 100, 500, 980)

  axis(1, at = c(-100, n_profs, 1e10), labels = c("", n_profs, ""), tck = -0.01)
  axis(2, at = seq(0,10), las = 1, tck = -0.01)

  # slight horizontal offsets so the markers don't overlap:
  offsets <- data.frame(pgdl = c(0.15, 0.5, 3, 7, 20, 30)) %>%
    mutate(dl = -pgdl, pb = 0, n_prof = n_profs)


  for (mod in c('pb','dl','pgdl')){
    mod_data <- filter(eval_data, model_type == mod)
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
  dev.off()
}

