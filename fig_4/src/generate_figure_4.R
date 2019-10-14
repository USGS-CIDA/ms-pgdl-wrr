
plot_figure_4 <- function(){
  library(dplyr)
  library(readr)
  library(stringr)
  library(sbtools) # see https://github.com/USGS-R/sbtools

  mendota_file <- tempfile('me_', fileext = '.csv')
  sparkling_file <- tempfile('sp_', fileext = '.csv')
  item_file_download('5d925066e4b0c4f70d0d0599', names = 'me_RMSE.csv', destinations = mendota_file)
  item_file_download('5d92507be4b0c4f70d0d059b', names = 'sp_RMSE.csv', destinations = sparkling_file)

  eval_data <- readr::read_csv(mendota_file, col_types = 'iccd') %>%
    filter(model_type == 'pgdl') %>%
    rbind(readr::read_csv(sparkling_file, col_types = 'iccd'))


    png(filename = 'figures/figure_4_wrr.png', width = 8, height = 4.5, units = 'in', res = 200)
    par(omi = c(0,0,0,0), mai = c(0.3,0.75,0.05,0.05), las = 1, mgp = c(2.5,.5,0), cex = 1)



  plot(NA, NA, xlim = c(0.8, 9.2), ylim = c(3.1, 0.8),
       ylab = 'Test RMSE (°C)', xlab = " ", axes = FALSE)
  gapper <- 0.09
  pt_cex <- 2

  n_prof <- c(2, 10, 50, 100, 500, 980, 500, 500, 500)
  xs <- 1:9
  for (type in c(rep('similar',7), 'year','season')){
    this_eval_data <- filter(eval_data, exper_id == sprintf('%s_%s', type, n_prof[1]))

    light_train <- filter(this_eval_data, model_type == 'pgdl_lim') %>% pull(rmse)
    libr_train <- filter(this_eval_data, model_type == 'pgdl') %>% pull(rmse)
    lines(x = c(xs[c(1,1)]-gapper), y = range(light_train), col = '#7570b3', lwd = 2.5)
    lines(x = c(xs[c(1,1)]+gapper), y = range(libr_train), col = '#7570b3', lwd = 2.5)

    light_train %>% mean %>% points(x = xs[1]-gapper, y = ., pch = 23, col = '#7570b3', bg = 'grey65', lwd = 2.5, cex = pt_cex, ljoin = 1)
    libr_train %>% mean %>% points(x = xs[1]+gapper, y = ., pch = 23, col = '#7570b3', bg = 'white', lwd = 2.5, cex = pt_cex, ljoin = 1)

    n_prof <- tail(n_prof, -1)
    xs <- tail(xs, -1)
  }


  axis(2, at = seq(0,10, 0.5), las = 1, tck = -0.01)
  par(mgp = c(1.5,.4,0))
  axis(1, at = c(-100, 1:9, 1e10), labels = c("", rev(c(expression("sim"['980']), expression("sim"['500']), expression("sim"['100']),
                                                        expression("sim"['50']), expression("sim"['10']), expression("sim"['2']))), expression("sim"['500']), expression("year"['500']), expression("seas"['500']), ""), tck = -0.01)

  abline(v = 6.5)
  text(6.5, 3.1, 'Experiments from Figure 3', pos = 4, cex = 0.7)
  text(6.5, 3.1, 'Experiments from Figure 2', pos = 2, cex = 0.7)

  points(0.8, 0.93, pch = 23, col = '#7570b3', bg = 'grey65', lwd = 2.5, cex = pt_cex, ljoin = 1)
  text(0.9, .94, 'PGDL limited pre-training', pos = 4)
  points(0.8, 0.78, pch = 23, col = '#7570b3', bg = 'white', lwd = 2.5, cex = pt_cex, ljoin = 1)
  text(0.9, 0.79, 'PGDL extended pre-training', pos = 4)



  dev.off()

}

generate_fig_5_text <- function(){

  if (type == 'seasons'){
    message("mendota seasons limited pre-train: ", light_train %>% mean %>% round(2))
    message("mendota seasons extended pre-train: ", libr_train %>% mean %>% round(2))
  }
  if (type == 'years'){
    message("mendota years limited pre-train: ", light_train %>% mean %>% round(2))
    message("mendota years extended pre-train: ", libr_train %>% mean %>% round(2))
  }
  if (type == 'years'){
    diff <- (light_train %>% mean %>% round(3)) - (libr_train %>% mean %>% round(3))
    message("mendota years diff: ", diff)
  }
  if (type == 'similar' & n_prof[1] == 2){
    message("mendota sparse limited pre-train: ", light_train %>% mean %>% round(2))
    message("mendota sparse extended pre-train: ", libr_train %>% mean %>% round(2))
  }
  if (type == 'similar'){
    diff <- (light_train %>% mean %>% round(2)) - (libr_train %>% mean %>% round(2))
    message("sim diff: ", n_prof[1], " ", diff)
  }


  years_msg <- "extended pre-training dataset were likely responsible for the %s°C RMSE improvement over predictions from the limited case (%s vs %s; Figure 5 year500), and for reducing the difference in accuracy between similar and years prediction challenges for Lake Mendota from X"
  light_train <- filter(data, n_profiles == 500, Model == "PGRNN", type == 'years') %>% pull(`Test RMSE`) %>% mean %>% round(2)
  ext_train <- filter(data, n_profiles == 500, Model == "PGRNN_pretrained_prev_yrs", type == 'years') %>% pull(`Test RMSE`) %>% mean %>% round(2)
  message(sprintf(years_msg, light_train-ext_train, ext_train, light_train))

  yr_msg2 <- "and for reducing the difference in accuracy between similar and years prediction challenges for Lake Mendota from %s (grey markers in Figure 5; %s vs %s..) to a negligible difference of %s (Figure 5 sim500 vs year500)"
  sim_light_train <- filter(data, n_profiles == 500, Model == "PGRNN", type == 'similar') %>% pull(`Test RMSE`) %>% mean %>% round(2)
  sim_ext_train <- filter(data, n_profiles == 500, Model == "PGRNN_pretrained_prev_yrs", type == 'similar') %>% pull(`Test RMSE`) %>% mean %>% round(2)
  message(sprintf(yr_msg2, light_train-sim_light_train, sim_light_train, light_train, ext_train - sim_ext_train))

}

plot_diamond <- function(x, y, width = 0.3, height = width, ...){
  x = c(x-width/2, x, x+width/2, x, x-width/2)
  y = c(y, y+height/2, y, y-height/2, y)
  polygon(x, y, ...)
}
