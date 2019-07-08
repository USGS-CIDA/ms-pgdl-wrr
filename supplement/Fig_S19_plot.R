plot_fig_s19 <- function(){
  library(ggplot2)

  # read in RMSE dat from fig 3
  source('supplement/Fig_S19_data_prep.R')
  rmse_long <- format_figS19_dat()

  mod_10 <- filter(rmse_long, model_10)
  mod_10$model <- factor(mod_10$model, levels = c('PB_10', "DL_10", "PGDL_10"))
  mod_other <- filter(rmse_long, !model_10)
  mod_other$model <- factor(mod_other$model, levels = c("PB_0", "PB_all","DL_all","PGDL_all"))

  p1 <- ggplot(mod_other, aes(x = lake_name, y = RMSE)) +
    geom_point(aes(color = model, shape = model, fill = model), size = 1.7, alpha = 0.8) +
    scale_shape_manual(values = c(1,21,22,23),
                       labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    scale_color_manual(values = c('#1b9e77','#1b9e77','#d95f02', '#7570b3'),
                       labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    scale_fill_manual(values = c('#1b9e77','#1b9e77','#d95f02', '#7570b3'),
                      labels = c(bquote(PB[0]), bquote(PB[all]),bquote(DL[all]), bquote(PGDL[all])))+
    #scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
    #facet_grid(rows = vars(model_10)) +
    theme_bw() +
    coord_cartesian(ylim = c(0.9, 5.2)) +
    theme(
      #axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = c(0.055,0.17),
      panel.grid.minor.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          legend.title = element_blank(),
          legend.background = element_rect(colour = 'black',
                                           fill = 'white', linetype='solid'),
          legend.text = element_text(size = 8),
          legend.key.height = unit(0.5, 'line'),
          axis.text.x = element_blank(),
          plot.margin = unit(c(0,0,0,0),'cm')) +
    labs(x = '') +
    scale_y_reverse()

  p2 <- ggplot(mod_10, aes(x = lake_name, y = RMSE)) +
    geom_point(aes(color = model, shape = model, fill = model), size = 1.7, alpha = 0.8) +
    scale_shape_manual(values = c(21,22,23),
                       labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    scale_color_manual(values = c('#1b9e77','#d95f02', '#7570b3'),
                       labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    scale_fill_manual(values = c('#1b9e77','#d95f02', '#7570b3'),
                      labels = c(bquote(PB[10]), bquote(DL[10]), bquote(PGDL[10]))) +
    #scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
    #facet_grid(rows = vars(model_10)) +
    theme_bw() +
    coord_cartesian(ylim = c(0.9, 5.2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          legend.position = c(0.055,0.16),
          panel.grid.minor.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          legend.title = element_blank(),
          legend.background = element_rect(colour = 'black',
                                           fill = 'white', linetype='solid'),
          legend.text = element_text(size = 8),
          legend.key.height = unit(0.5, 'line'),
          plot.margin = unit(c(0,0,0,0),'cm')) +
    labs(x = '') +
    scale_y_reverse()

  p <- cowplot::plot_grid(p1, p2, ncol = 1, rel_heights = c(0.4, 0.56))

  dir.create('supplement/out', showWarnings = FALSE)

  ggsave('supplement/out/supplement_fig_S19.png', p,
         height = 7, width = 10, units = 'in')

  # upload to supplement folder
  drive_update(file = as_id('1yCCcqfPeppdQM79adK5dWrUmMzIwBvQv'),
               media = 'supplement/out/supplement_fig_S19.png')
}
