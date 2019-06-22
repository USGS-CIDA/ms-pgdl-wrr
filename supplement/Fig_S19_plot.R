plot_fig_s19 <- function(){
  library(ggplot2)

  # read in RMSE dat from fig 3
  source('supplement/Fig_S19_data_prep.R')
  rmse_long <- format_figS19_dat()

  p <- ggplot(rmse_long, aes(x = lake_name, y = RMSE)) +
    geom_point(aes(color = model, shape = model, fill = model), size = 1.7) +
    scale_shape_manual(values = c(22,0,21,1,23,5))+
    scale_color_manual(values = c('#d95f02', '#d95f02', '#1b9e77','#1b9e77', '#7570b3','#7570b3')) + # match Jordan's colors in fig 3
    scale_fill_manual(values = c('#d95f02', '#d95f02', '#1b9e77','#1b9e77', '#7570b3','#7570b3')) + # match Jordan's colors in fig 3
    #scale_shape_manual(values = c(16, 21, 23)) + # match Jordan's shapes in fig 3
    theme_bw() +
    coord_cartesian(ylim = c(0.9, 7.5)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
          legend.position = c(0.1,0.15),
          panel.grid.minor.y = element_blank(),
          panel.grid.minor.x = element_blank(),
          legend.title = element_blank(),
          legend.background = element_rect(colour = 'black',
                                           fill = 'white', linetype='solid'),
          legend.text = element_text(size = 8),
          legend.key.height = unit(0.5, 'line')) +
    labs(x = '') +
    scale_y_reverse()

  dir.create('supplement/out', showWarnings = FALSE)

  ggsave('supplement/out/supplement_fig_S19.png', p,
         height = 150, width = 230, units = 'mm', scale = 0.9)

  # upload to supplement folder
  drive_update(file = as_id('https://drive.google.com/drive/u/1/folders/1yCCcqfPeppdQM79adK5dWrUmMzIwBvQv'),
               media = 'supplement/out/supplement_fig_S19.png')
}
