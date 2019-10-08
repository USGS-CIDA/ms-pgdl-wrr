plot_seasons_years_sparsity <- function(){
  library(dplyr)

  library(dplyr)
  library(readr)
  library(stringr)

  eval_data <- readr::read_csv('data_release/out/me_RMSE.csv', col_types = 'iccd') %>%
    mutate(lake_id = 'M') %>%
    rbind(readr::read_csv('data_release/out/sp_RMSE.csv', col_types = 'iccd') %>% mutate(lake_id = 'S')) %>%
    filter(str_detect(exper_id, '[a-z]+_500')) %>%
    mutate(col = case_when(
      exper_model == 'pb' ~ '#1b9e77',
      exper_model == 'dl' ~'#d95f02',
      exper_model == 'pgdl' ~ '#7570b3'
    ), pch = case_when(
      exper_model == 'pb' ~ 21,
      exper_model == 'dl' ~ 22,
      exper_model == 'pgdl' ~ 23
    ), x_pos = case_when(
      exper_model == 'pb' ~ 1,
      exper_model == 'dl' ~ 2,
      exper_model == 'pgdl' ~ 3
    ), x_bmp = case_when(
      lake_id == 'S' ~ 0.1,
      lake_id == 'M' ~ -0.1
    ), cex = case_when(
      exper_model == 'dl' ~ 3.5,
      TRUE ~ 3.2
    ))

  png(filename = 'figures/figure_2_wrr.png', width = 8, height = 4.5, units = 'in', res = 200)
  par(omi = c(0,0.4,0.05,0.25), mai = c(0.5,0.2,0,0), las = 1, mgp = c(2.3,.5,0))
  layout(mat = matrix(c(1,2,3), nrow = 1))

  set_plot <- function(text, panel){
    plot(NA, NA, xlim = c(0.7, 3.3), ylim = c(3.0, .8),
         ylab = '', xlab = "", axes = FALSE, xaxs = 'i', yaxs = 'i')
    if (panel == 'a)'){
      axis(2, at = seq(0,10, by = 0.5), las = 1, tck = -0.01, cex.axis = 1.4)
      mtext(text = 'Test RMSE (°C)', side = 2, las = 3, outer = 3, padj = -1.5)
    } else {
      axis(2, at = seq(0,10, by = 0.5), labels = NA, las = 1, tck = -0.01)
    }
    axis(1, at = c(0,1,2,3,4), labels = c("", 'Process-', 'Deep', 'Process-Guided', ""), tck = -0.01)
    par(mgp = c(2.3,1.5,0))
    axis(1, at = c(0,1,2,3,4), labels = c('', "Based", 'Learning', 'Deep Learning', ""), tck = 0)
    par(mgp = c(2.3,0.5,0))
    text(x = 0.75, y = 0.89, panel, font = 2, pos = 4, cex = 1.4)
    if (length(text) == 1){
      text(x = 0.95, y = 0.89, text, pos = 4, cex = 1.2)
    } else {
      text(x = 0.95, y = 0.84, text[1], pos = 4, cex = 1.2)
      text(x = 0.95, y = 0.94, text[2], pos = 4, cex = 1.2)
    }

  }


  plot_all_models <- function(plot_data){

    for (mod in c('pb','dl','pgdl')){
      mod_data <- filter(plot_data, exper_model == mod)
      for (lake_id in c('S','M')){

        ._d <- filter(mod_data, lake_id == !!lake_id) %>%
          summarize(y0 = min(rmse), y1 = max(rmse), y = mean(rmse),
                    x0 = mean(x_pos + x_bmp), x1 = mean(x_pos + x_bmp),
                    col = head(col,1), pch = head(pch,1), cex = head(cex,1))
        lines(c(._d[c('x0','x1')]), c(._d[c('y0','y1')]), col = ._d$col, lwd = 2.5)
        points(._d$x0, ._d$y, pch = ._d$pch, lwd = 1.5, cex = ._d$cex, bg = 'white', col = ._d$col, ljoin = 1)
        text(._d$x0, ._d$y, lake_id, font = 2)
      }
    }
  }

  set_plot("Train & test similar", "a)")
  plot_all_models(filter(eval_data, str_detect(exper_id, 'similar_[0-9]+')))

  set_plot(c("Train: coldest years","Test: warmest years"), "b)")
  plot_all_models(filter(eval_data, str_detect(exper_id, 'year_[0-9]+')))


  set_plot(c("Train: spring, fall, winter","Test: summer"), "c)")
  plot_all_models(filter(eval_data, str_detect(exper_id, 'season_[0-9]+')))
  dev.off()

}
