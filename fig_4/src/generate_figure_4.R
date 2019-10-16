
plot_figure_4 <- function(){
  library(dplyr)
  library(readr)
  library(stringr)
  library(sbtools) # see https://github.com/USGS-R/sbtools

  mendota_file <- tempfile('me_', fileext = '.csv')
  mendota_lim_file <- tempfile('me_lim_', fileext = '.csv')
  item_file_download('5d925066e4b0c4f70d0d0599', names = 'me_RMSE.csv', destinations = mendota_file)
  item_file_download('5d925066e4b0c4f70d0d0599', names = 'me_RMSE_limited_training.csv', destinations = mendota_lim_file)

  eval_data <- readr::read_csv(mendota_file, col_types = 'iccd') %>%
    filter(model_type == 'pgdl') %>%
    rbind(readr::read_csv(mendota_lim_file, col_types = 'iccd'))


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
  return(eval_data)
}

generate_text_fig_4 <- function(eval_data){
  render_data <- list(prof_2_lim = filter(eval_data, exper_id  == 'similar_2', model_type == 'pgdl_lim') %>% pull(rmse) %>% mean() %>% round(2),
                      prof_2_ext = filter(eval_data, exper_id  == 'similar_2', model_type == 'pgdl') %>% pull(rmse) %>% mean() %>% round(2),
                      sea_lim = filter(eval_data, exper_id  == 'season_500', model_type == 'pgdl_lim') %>% pull(rmse) %>% mean() %>% round(2),
                      sea_ext = filter(eval_data, exper_id  == 'season_500', model_type == 'pgdl') %>% pull(rmse) %>% mean() %>% round(2),
                      yr_lim = filter(eval_data, exper_id  == 'year_500', model_type == 'pgdl_lim') %>% pull(rmse) %>% mean() %>% round(2),
                      yr_ext = filter(eval_data, exper_id  == 'year_500', model_type == 'pgdl') %>% pull(rmse) %>% mean() %>% round(2),
                      yr_dif = filter(eval_data, exper_id  == 'year_500') %>% group_by(model_type) %>% summarize(rmse = mean(rmse)) %>% pull(rmse) %>% diff() %>% round(2),
                      yr_lim_dif = filter(eval_data, exper_id  %in% c('similar_500', 'year_500'), model_type == 'pgdl_lim') %>% group_by(exper_id) %>% summarize(rmse = mean(rmse)) %>% pull(rmse) %>% diff() %>% round(2),
                      sim_ext = filter(eval_data, exper_id  == 'similar_500', model_type == 'pgdl') %>% pull(rmse) %>% mean() %>% round(2))

  template <- 'The two largest differences in prediction accuracy between the extended and limited pre-training datasets were when only two profiles were used for training ({{prof_2_ext}} vs {{prof_2_lim}}°C RMSE; ****
  and when models were trained using data from colder seasons and used to predict summer temperatures ({{sea_ext}} vs {{sea_lim}}°C RMSE; Figure 5 *****
  Excluding summer data from pre-training decreased PGDL performance and resulted in worse temperature estimates than the calibrated PB model (as measured by RMSE; {{sea_lim}} vs 1.91°C, respectively; Figure 3c ******
  When models were trained on colder years and used to predict warmer years, the additional complete years included in the
  extended pre-training dataset were likely responsible for the {{yr_dif}}°C RMSE improvement over predictions from the limited case ({{yr_ext}} vs {{yr_lim}}°C, respectively; Figure 5 ******
  and for reducing the difference in accuracy between similar and years prediction challenges for Lake Mendota from {{yr_lim_dif}}°C to
  a negligible difference (difference between grey filled diamonds for similar___500 ({{sim_ext}}°C) and year___500 ({{yr_ext}}°C) '
  whisker::whisker.render(template %>% str_remove_all('\n') %>% str_replace_all('  ', ' '), render_data ) %>% cat()
}
