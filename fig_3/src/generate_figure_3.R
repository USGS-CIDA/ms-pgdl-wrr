

plot_calibrated_figure_3 <- function(){

  library(dplyr)
  library(readr)
  library(tidyr)
  library(beanplot)

  eval_data <- readr::read_csv('data_release/out/all_RMSE.csv', col_types = 'icccd')

  png(filename = 'figures/figure_3_wrr.png', width = 7, height = 7, units = 'in', res = 200)
  par(omi = c(0.6,0,0.1,0.2), mai = c(0.2,0.8,0,0), las = 1, mgp = c(2.2,0.8,0))

  ylim <- c(5.5, 0.5)
  xlim <- c(0.7, 4.5)

  flt <- function(data, model_type){
    filter(data, model_type == !!model_type) %>%
      pull(rmse)
  }

  positions = c(pb0 = 1, pb = 2, dl = 3, pgdl = 4)

  plot(NA, NA, xlim = xlim, ylim = ylim,
       ylab = 'Test RMSE (°C)', axes = FALSE, xaxs = 'i', yaxs = 'i', cex.lab = 1.5)

  bean_w <- 0.65

  beanplot(flt(eval_data, 'pb0'), flt(eval_data, 'pb'), flt(eval_data, 'dl'), flt(eval_data, 'pgdl'), maxwidth = bean_w, what=c(0,1,0,0), log = "", add = TRUE,
            axes = F, border = NA, at = positions, col = list('grey65','#1b9e77','#d95f02','#7570b3'))

  med_w <- 0.4
  ind_w <- 0.03
  for (model in c('pb0', 'pb', 'dl', 'pgdl')){
    segments(x0 = positions[[model]]-ind_w, x1 = positions[[model]]+ind_w, y0 = flt(eval_data, model), col = 'black')
    segments(x0 = positions[[model]]-med_w/2, x1 = positions[[model]]+med_w/2, y0 = median(flt(eval_data, model)), col = 'black', lwd = 2)
  }

  axis(2, at = seq(0,10), las = 1, tck = -0.01, cex.axis = 1.5, lwd = 1.5)


  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", "Process-", "Process-", "Deep", "Processs-Guided", ""),  tck = -0.01, cex.axis = 1.5, lwd = 1.5)
  par(mai = c(0,0,0,0), mgp = c(2, 1.5,0))
  axis(1, at = c(-100, 1, 2, 3, 4, 1e10), labels = c("", expression("Based"['uncal']), "Based", "Learning","Deep Learning", ""), tck = NA, cex.axis = 1.5, lwd = NA)

  dev.off()
}


generate_fig_4_text <- function(){
  sprintf('Predictions from PGDL models applied to %s lakes were more accurate or as accurate (within +/-0.05°C RMSE) as all but %s of the calibrated PB models and %s DL models (',
          n_sims,
          sum(plot_data$PGDL_all > plot_data$PB_all+0.05),
          sum(plot_data$PGDL_all > plot_data$DL_all+0.05)) %>% message()

  sprintf('The median RMSE (across all lakes) was %s° for the PGDL model, %s°C for PB, and %s°C for DL. ',
          round(median(plot_data[['PGDL_all']]),2),
          round(median(plot_data[['PB_all']]),2),
          round(median(plot_data[['DL_all']]),2)) %>% message()

  sprintf('The range of prediction accuracy for PGDL models was %s to %s°C, %s to %s°C for PB, and %s to %s°C for DL. ',
          round(range(plot_data$PGDL_all)[1], 2), round(range(plot_data$PGDL_all)[2], 2),
          round(range(plot_data$PB_all)[1], 2), round(range(plot_data$PB_all)[2], 2),
          round(range(plot_data$DL_all)[1], 2), round(range(plot_data$DL_all)[2], 2)) %>%
    message()

  sprintf('The uncalibrated GLM predictions (which are used for pre-training PGDL) had a median RMSE of %s°C, with a range of %s to %s°C.',
          round(median(plot_data[['PB_uncal']]),2),
          round(range(plot_data[['PB_uncal']])[1],2),
          round(range(plot_data[['PB_uncal']])[2],2)) %>%
    message()

  sprintf('The uncalibrated GLM predictions were worse than PB in %s lakes (%s%% of total)',
          sum(plot_data$PB_uncal > plot_data$PB_all),
          round((sum(plot_data$PB_uncal > plot_data$PB_all)/n_sims)*100, 1)) %>%
    message()

  sprintf('were variable across lakes, with %s lakes improving RMSE by over 2° compared to pre-trainer RMSEs, while %s lakes had PGDL predictions that were approximately equal to the pre-trainer (range of %s to %s°C increase in RMSE). ',
          sum(plot_data$PGDL_all + 2.0 < plot_data$PB_uncal),
          sum(plot_data$PGDL_all > plot_data$PB_uncal + 0.05 | plot_data$PGDL_all > plot_data$PB_uncal - 0.05),
          round(range(filter(plot_data, PGDL_all - PB_uncal > 0 ) %>% mutate(diff = PGDL_all - PB_uncal) %>% pull(diff))[1],3),
          round(range(filter(plot_data, PGDL_all - PB_uncal > 0 ) %>% mutate(diff = PGDL_all - PB_uncal) %>% pull(diff))[2],3)) %>%
    message()

  sprintf('The difference in RMSE between PB and PGDL ranged from %s to %s°C and %s to %s°C for DL to PGDL',
          round(range(plot_data$PB_all - plot_data$PGDL_all)[1], 2), round(range(plot_data$PB_all - plot_data$PGDL_all)[2], 2),
          round(range(plot_data$DL_all - plot_data$PGDL_all)[1], 2), round(range(plot_data$DL_all - plot_data$PGDL_all)[2], 2)) %>%
    message()

  sprintf('When observations were artificially removed to leave only 10 dates for training, predictions from PGDL models were more accurate than %s of the calibrated PB models (%s%% of total) and more accurate than %s DL models (Figure 4).',
          sum(plot_data$PB_10 > plot_data$PGDL_10 - 0.05),
          round((sum(plot_data$PB_10 > plot_data$PGDL_10 - 0.05)/n_sims)*100, 1),
          ifelse(all(plot_data$DL_10 > plot_data$PGDL_10 - 0.05), 'all', sum(plot_data$DL_10 > plot_data$PGDL_10 - 0.05))) %>% message()

  sprintf('The median RMSE (across all lakes) for 10 training profiles was %s° for PGDL, %s°C for PB, and %s°C for DL. ',
          round(median(plot_data[['PGDL_10']]),2),
          round(median(plot_data[['PB_10']]),2),
          round(median(plot_data[['DL_10']]),2)) %>% message()

  sprintf('With the reduced training dataset, %s lakes had worse PGDL predictions compared to the pre-trainer (range of %s to %s°C increase in RMSE) while %s lakes with PGDL models decreased prediction RMSE by more than 1° compared to the pre-trainer (greatest improvement was a reduction of %s°C RMSE). ',
          sum(plot_data$PGDL_10 > plot_data$PB_uncal),
          round(range(filter(plot_data, PGDL_10 - PB_uncal > 0 ) %>% mutate(diff = PGDL_10 - PB_uncal) %>% pull(diff))[1],3),
          round(range(filter(plot_data, PGDL_10 - PB_uncal > 0 ) %>% mutate(diff = PGDL_10 - PB_uncal) %>% pull(diff))[2],3),
          sum(plot_data$PGDL_10 + 1.0 < plot_data$PB_uncal),
          round(max(plot_data$PB_uncal- plot_data$PGDL_10), 2)) %>%
    message()
}
