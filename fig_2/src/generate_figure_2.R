plot_figure_2 <- function(){
  library(dplyr)
  library(readr)
  library(stringr)

  library(sbtools) # see https://github.com/USGS-R/sbtools

  mendota_file <- tempfile('me_', fileext = '.csv')
  sparkling_file <- tempfile('sp_', fileext = '.csv')
  item_file_download('5d925066e4b0c4f70d0d0599', names = 'me_RMSE.csv', destinations = mendota_file)
  item_file_download('5d92507be4b0c4f70d0d059b', names = 'sp_RMSE.csv', destinations = sparkling_file)

  eval_data <- readr::read_csv(mendota_file, col_types = 'iccd') %>%
    mutate(lake_id = 'M') %>%
    rbind(readr::read_csv(sparkling_file, col_types = 'iccd') %>% mutate(lake_id = 'S')) %>%
    filter(str_detect(exper_id, '[a-z]+_500')) %>%
    mutate(col = case_when(
      model_type == 'pb' ~ '#1b9e77',
      model_type == 'dl' ~'#d95f02',
      model_type == 'pgdl' ~ '#7570b3'
    ), pch = case_when(
      model_type == 'pb' ~ 21,
      model_type == 'dl' ~ 22,
      model_type == 'pgdl' ~ 23
    ), x_pos = case_when(
      model_type == 'pb' ~ 1,
      model_type == 'dl' ~ 2,
      model_type == 'pgdl' ~ 3
    ), x_bmp = case_when(
      lake_id == 'S' ~ 0.1,
      lake_id == 'M' ~ -0.1
    ), cex = case_when(
      model_type == 'dl' ~ 3.5,
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
      mod_data <- filter(plot_data, model_type == mod)
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
  return(eval_data)
}


generate_text_fig_2 <- function(eval_data){

  render_data <- list(pb_year_sp = filter(eval_data, model_type == 'pb', exper_id == "year_500", lake_id == 'S') %>% pull(rmse) %>% mean %>% round(2),
                      pgdl_year_sp = filter(eval_data, model_type == 'pgdl', exper_id == "year_500", lake_id == 'S') %>% pull(rmse) %>% mean %>% round(2),
                      pgdl_year_me = filter(eval_data, model_type == 'pb', exper_id == "year_500", lake_id == 'M') %>% pull(rmse) %>% mean %>% round(2),
                      pgdl_sim_me = filter(eval_data, model_type == 'pb', exper_id == "similar_500", lake_id == 'M') %>% pull(rmse) %>% mean %>% round(2),
                      pb_sim_sp = filter(eval_data, model_type == 'pb', exper_id == "similar_500", lake_id == 'S') %>% pull(rmse) %>% mean %>% round(2),
                      pb_year_me = filter(eval_data, model_type == 'pb', exper_id == "year_500", lake_id == 'M') %>% pull(rmse) %>% mean %>% round(2),
                      dl_year_me = filter(eval_data, model_type == 'dl', exper_id == "year_500", lake_id == 'M') %>% pull(rmse) %>% mean %>% round(2))

  template <- 'experiment, when the 3 warmest years were withheld from model construction ({{pb_year_sp}} and {{pgdl_year_sp}}°C RMSE for PB and PDGL, respectively).
  In-bounds predictions (Figure 3a) were generally more accurate for all three modeling approaches compared to out-of-bounds predictions (Figure 3b; 3c),
  with both exceptions appearing in the years predictions. Lake Mendota’s PGDL years predictions were approximately the same accuracy as the
  in-bounds prediction mean ({{pgdl_year_me}} and {{pgdl_sim_me}}°C), while Sparkling Lake’s PB years predictions were an improvement over in-bounds ({{pb_year_sp}} and {{pb_sim_sp}}°C).
  Process-based models were more accurate in their out-of-bounds predictions than DL models except for Lake Mendota in the years experiment ({{pb_year_me}}°C for PB and {{dl_year_me}}°C for DL;'

  whisker::whisker.render(template %>% str_remove_all('\n') %>% str_replace_all('  ', ' '), render_data ) %>% cat()



}
