plot_seasons_years_sparsity <- function(){
  library(dplyr)

  data <- readr::read_csv('~/Downloads/revision_Figure_2_results - Sheet1 (7).csv')#'Figure 2 seasonal_yearly results - Sheet1 (2).csvfig_2/in/Figure 2 seasonal_yearly results - Sheet1.csv')


  png(filename = 'figures/figure_2_wrr.png', width = 8, height = 4.5, units = 'in', res = 200)
  par(omi = c(0,0.4,0.05,0.25), mai = c(0.5,0.2,0,0), las = 1, mgp = c(2.3,.5,0))
  layout(mat = matrix(c(1,2,3), nrow = 1))

  set_plot <- function(text, panel){
    plot(NA, NA, xlim = c(0.7, 3.3), ylim = c(3.3, .8),
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

  sp_bmp <- 0.1
  me_bmp <- -0.1
  RNN_x <- 2
  PGRNN_x <- 3
  GLM_x <- 1

  set_plot("Train & test similar", "a)")

  plot_data <- data %>% filter(substr(Experiment, 1, 7) == 'similar') %>% mutate(lake = substr(Experiment, 12, length(Experiment)))

  plot_all_models <- function(plot_data){
    lines(c(RNN_x+me_bmp, RNN_x+me_bmp), c(filter(plot_data, lake == "mendota", Model == 'RNN') %>% pull(`Test RMSE`) %>% max(),
                                           filter(plot_data, lake == "mendota", Model == 'RNN') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#d95f02')
    points(RNN_x+me_bmp, filter(plot_data, lake == "mendota", Model == 'RNN') %>% pull(`Test RMSE`) %>% mean, pch = 22, lwd = 1.5, cex = 3.5, bg = 'white', col = '#d95f02', ljoin = 1)
    text(RNN_x+me_bmp, filter(plot_data, lake == "mendota", Model == 'RNN') %>% pull(`Test RMSE`) %>% mean, "M", font = 2)
    message("Mendota RNN:", filter(plot_data, lake == "mendota", Model == 'RNN') %>% pull(`Test RMSE`) %>% mean)

    ME_mean <- filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% mean
    message("Mendota PGRNN: ", round(ME_mean, 2))
    lines(c(PGRNN_x+me_bmp, PGRNN_x+me_bmp), c(filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% max(),
                                               filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#7570b3')
    points(PGRNN_x+me_bmp, ME_mean, pch = 23, lwd = 1.5, cex = 3.2, bg = 'white', col = '#7570b3', ljoin = 1)
    text(PGRNN_x+me_bmp, ME_mean, "M", font = 2)

    # ME_mean <- filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% mean
    # lines(c(PGRNN_x+me_bmp-0.05, PGRNN_x+me_bmp-0.05), c(filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% max(),
    #                                            filter(plot_data, lake == "mendota", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#FF007f')
    # points(PGRNN_x+me_bmp-0.05, ME_mean, pch = 23, lwd = 1.5, cex = 3.2, bg = 'white', col = '#FF007f')
    # text(PGRNN_x+me_bmp-0.05, ME_mean, "M", font = 2)



    ME_mean <- filter(plot_data, lake == "mendota", Model == 'GLM') %>% pull(`Test RMSE`) %>% mean
    lines(c(GLM_x+me_bmp, GLM_x+me_bmp), c(filter(plot_data, lake == "mendota", Model == 'GLM') %>% pull(`Test RMSE`) %>% max(),
                                           filter(plot_data, lake == "mendota", Model == 'GLM') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#1b9e77')
    points(GLM_x+me_bmp, ME_mean, pch = 21, lwd = 1.5, cex = 3.2, bg = 'white', col = '#1b9e77', ljoin = 1)
    text(GLM_x+me_bmp, ME_mean, "M", font = 2)
    message("Mendota GLM:", round(ME_mean,2))

    SP_mean <- filter(plot_data, lake == "sparkling", Model == 'RNN') %>% pull(`Test RMSE`) %>% mean
    lines(c(RNN_x+sp_bmp, RNN_x+sp_bmp), c(filter(plot_data, lake == "sparkling", Model == 'RNN') %>% pull(`Test RMSE`) %>% max(),
                                           filter(plot_data, lake == "sparkling", Model == 'RNN') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#d95f02')
    points(RNN_x+sp_bmp, SP_mean, pch = 22, lwd = 1.5, cex = 3.5, bg = 'white', col = '#d95f02', ljoin = 1)
    text(RNN_x+sp_bmp, SP_mean, "S", font = 2)

    SP_mean <- filter(plot_data, lake == "sparkling", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% mean
    lines(c(PGRNN_x+sp_bmp, PGRNN_x+sp_bmp), c(filter(plot_data, lake == "sparkling", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% max(),
                                               filter(plot_data, lake == "sparkling", Model == 'PGRNN_pretrained_prev_yrs') %>% pull(`Test RMSE`) %>% min()), lwd = 1.5, col = '#7570b3')
    points(PGRNN_x+sp_bmp, SP_mean, pch = 23, lwd = 1.5, cex = 3.2, bg = 'white', col = '#7570b3', ljoin = 1)
    text(PGRNN_x+sp_bmp, SP_mean, "S", font = 2)

    SP_mean <- filter(plot_data, lake == "sparkling", Model == 'GLM') %>% pull(`Test RMSE`) %>% mean(na.rm = T)
    lines(c(GLM_x+sp_bmp, GLM_x+sp_bmp), c(filter(plot_data, lake == "sparkling", Model == 'GLM') %>% pull(`Test RMSE`) %>% max(na.rm = T),
                                           filter(plot_data, lake == "sparkling", Model == 'GLM') %>% pull(`Test RMSE`) %>% min(na.rm = T)), lwd = 1.5, col = '#1b9e77')
    points(GLM_x+sp_bmp, SP_mean, pch = 21, lwd = 1.5, cex = 3.2, bg = 'white', col = '#1b9e77', ljoin = 1)
    text(GLM_x+sp_bmp, SP_mean, "S", font = 2)


    message("Sparkling GLM:", SP_mean)
  }

  plot_all_models(plot_data)

  set_plot(c("Train: coldest years","Test: warmest years"), "b)")

  plot_data <- data %>% filter(substr(Experiment, 1, 4) == 'year') %>% mutate(lake = substr(Experiment, 10, length(Experiment)))

  plot_all_models(plot_data)

  set_plot(c("Train: spring, fall, winter","Test: summer"), "c)")
  # seasons here...
  plot_data <- data %>% filter(substr(Experiment, 1, 4) == 'seas') %>% mutate(lake = substr(Experiment, 12, length(Experiment)))

  plot_all_models(plot_data)

  dev.off()

}
