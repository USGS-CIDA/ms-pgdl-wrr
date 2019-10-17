

plot_figure_3 <- function(){

  library(dplyr)
  library(readr)
  library(tidyr)
  library(beanplot)
  library(sbtools) # see https://github.com/USGS-R/sbtools

  all_68_file <- tempfile('68_', fileext = '.csv')
  item_file_download('5d925048e4b0c4f70d0d0596', names = 'all_RMSE.csv', destinations = all_68_file, overwrite_file = TRUE)

  eval_data <- readr::read_csv(all_68_file, col_types = 'icccd') %>%
    group_by(exper_id, site_id, model_type) %>% summarize(rmse = median(rmse))

  pdf(file = 'figures/figure_3_wrr.pdf', width = 7, height = 7)
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
  return(eval_data)
}


generate_text_fig_3 <- function(eval_data){

  #Predictions from PGDL models applied to 68 lakes were more accurate or as accurate (within +/-0.05°C RMSE)

  render_data <- list(pb_better_pgdl = filter(eval_data, model_type %in% c('pgdl','pb')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pgdl-pb) %>% pull(dif) %>% {. > 0.05} %>% sum,
                      dl_better_pgdl = filter(eval_data, model_type %in% c('pgdl','dl')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pgdl-dl) %>% pull(dif) %>% {. > 0.05} %>% sum,
                      pgdl_2better_pb0 = filter(eval_data, model_type %in% c('pgdl','pb0')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pb0-pgdl) %>% pull(dif) %>% {. > 2} %>% sum,
                      pgdl_sbetter_pb0 = filter(eval_data, model_type %in% c('pgdl','pb0')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pb0-pgdl) %>% pull(dif) %>% {{0.05 < .} & {. <= 2}} %>% sum,
                      pgdl_same_pb0 = filter(eval_data, model_type %in% c('pgdl','pb0')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pb0-pgdl) %>% pull(dif) %>% {0.05 >= .} %>% sum,
                      pb_pgdl_min = filter(eval_data, model_type %in% c('pgdl','pb')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pb-pgdl) %>% pull(dif) %>% min %>% round(2),
                      pb_pgdl_max = filter(eval_data, model_type %in% c('pgdl','pb')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = pb-pgdl) %>% pull(dif) %>% max %>% round(2),
                      dl_pgdl_min = filter(eval_data, model_type %in% c('pgdl','dl')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = dl-pgdl) %>% pull(dif) %>% min %>% round(2),
                      dl_pgdl_max = filter(eval_data, model_type %in% c('pgdl','dl')) %>% select(site_id, model_type, rmse) %>% spread(model_type, rmse) %>% mutate(dif = dl-pgdl) %>% pull(dif) %>% max %>% round(2),
                      pgdl_median = filter(eval_data, model_type == 'pgdl') %>% pull(rmse) %>% median %>% round(2),
                      dl_median = filter(eval_data, model_type == 'dl') %>% pull(rmse) %>% median %>% round(2),
                      pb_median = filter(eval_data, model_type == 'pb') %>% pull(rmse) %>% median %>% round(2),
                      pb0_median = filter(eval_data, model_type == 'pb0') %>% pull(rmse) %>% median %>% round(2),
                      pgdl_min = filter(eval_data, model_type == 'pgdl') %>% pull(rmse) %>% min %>% round(2),
                      pgdl_max = filter(eval_data, model_type == 'pgdl') %>% pull(rmse) %>% max %>% round(2),
                      dl_min = filter(eval_data, model_type == 'dl') %>% pull(rmse) %>% min %>% round(2),
                      dl_max = filter(eval_data, model_type == 'dl') %>% pull(rmse) %>% max %>% round(2),
                      pb_min = filter(eval_data, model_type == 'pb') %>% pull(rmse) %>% min %>% round(2),
                      pb_max = filter(eval_data, model_type == 'pb') %>% pull(rmse) %>% max %>% round(2),
                      pb0_min = filter(eval_data, model_type == 'pb0') %>% pull(rmse) %>% min %>% round(2),
                      pb0_max = filter(eval_data, model_type == 'pb0') %>% pull(rmse) %>% max %>% round(2))

  template <- 'as all but {{pb_better_pgdl}} of the calibrated PB models and {{dl_better_pgdl}} of the DL models (Figure 4; see PGDL, PB, and DL and Supplement Table S3 and Figure S18
  for detailed lake-specific results; all RMSE values reported here correspond to model performance in the test period).
  The median RMSE (across all lakes) was {{pgdl_median}}°C for PGDL, {{dl_median}}°C for DL, and {{pb_median}}°C for PB.
  The range of prediction accuracy for PGDL models was {{pgdl_min}} to {{pgdl_max}}°C, {{dl_min}} to {{dl_max}}°C for DL, and {{pb_min}} to {{pb_max}}°C for PB. ***********
  which were used to pre-train PGDL) had a median RMSE of {{pb0_median}}°C, with a range of {{pb0_min}} to {{pb0_max}}°C ***********
  predictions used for pre-training were variable across lakes, with {{pgdl_2better_pb0}} lakes improving RMSE by over 2° compared to pre-trainer RMSEs,
  {{pgdl_sbetter_pb0}} lakes improving by smaller amounts, and {{pgdl_same_pb0}} lakes with PGDL prediction accuracy that was approximately equal to the pre-trainer accuracy in the test period ***********
  When comparing performance of predictions on individual lakes, the difference in RMSE between PB and PGDL ranged from {{pb_pgdl_min}} to {{pb_pgdl_max}}°C
  and {{dl_pgdl_min}} to {{dl_pgdl_max}}°C for DL to PGDL (positive values indicate better performance by PGDL; see also Supplement Table S3)'

  whisker::whisker.render(template %>% str_remove_all('\n') %>% str_replace_all('  ', ' '), render_data ) %>% cat()
}
