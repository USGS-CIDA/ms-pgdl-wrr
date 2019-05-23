
plot_data_sparsity <- function(){
  library(dplyr)
  data <- readr::read_csv('~/Downloads/Sparse training results (2_prof_GLM-uncal) - Sheet1.csv')#'fig_1/in/Sparse training results.csv')


  png(filename = 'figures/figure_1_wrr.png', width = 8, height = 10, units = 'in', res = 200)
  par(omi = c(0,0,0.05,0.05), mai = c(1,1,0,0), las = 1, mgp = c(2,.5,0), cex = 1.5)

  plot(NA, NA, xlim = c(2, 1000), ylim = c(4, 0.75),
       ylab = 'Test RMSE (°C)', xlab = "Training temperature profiles (#)", log = 'x', axes = FALSE)

  axis(1, at = c(-100,2, 10, 50, 100, 980, 1e10), labels = c("", 2, 10, 50, 100, 980, ""), tck = -0.01)
  axis(2, at = seq(0,10), las = 1, tck = -0.01)

  pgrnn_offsets <- c(0.15, 0.5, 3, 7, 30)
  rnn_offsets <- -pgrnn_offsets

  pg_mean <- list(x = c(), y = c(), train = c())
  pgto_mean <- list(x = c(), y = c(), train = c())
  rnn_mean <- list(x = c(), y = c(), train = c())
  glm_mean <- list(x = c(), y = c(), train = c())

  for (x in c(2, 10, 50, 100, 980)){
    this_mean <- filter(data, n_profiles == x, Model == "GLM") %>% pull(`Test RMSE`) %>% mean
    this_train <- filter(data, n_profiles == x, Model == "GLM") %>% pull(`Train RMSE`) %>% mean
    glm_mean$train <- c(glm_mean$train, this_train)
    glm_mean$y <- c(glm_mean$y, this_mean)

    glm_mean$x <- c(glm_mean$x, x)
    lines(c(x, x), c(filter(data, n_profiles == x, Model == "GLM") %>% pull(`Test RMSE`) %>% max(),
                     filter(data, n_profiles == x, Model == "GLM") %>% pull(`Test RMSE`) %>% min()), col = '#1b9e77', lwd = 2.5)


    this_mean <- filter(data, n_profiles == x, Model == "RNN") %>% pull(`Test RMSE`) %>% mean
    this_train <- filter(data, n_profiles == x, Model == "RNN") %>% pull(`Train RMSE`) %>% mean
    rnn_mean$train <- c(rnn_mean$train, this_train)
    rnn_mean$y <- c(rnn_mean$y, this_mean)
    rnn_mean$x <- c(rnn_mean$x, x+rnn_offsets[1])
    lines(c(x+rnn_offsets[1], x+rnn_offsets[1]), c(filter(data, n_profiles == x, Model == "RNN") %>% pull(`Test RMSE`) %>% max(),
                                                   filter(data, n_profiles == x, Model == "RNN") %>% pull(`Test RMSE`) %>% min()), col = '#d95f02', lwd = 2.5)

    this_mean <- filter(data, n_profiles == x, Model == "PGRNN") %>% pull(`Test RMSE`) %>% mean
    this_train <- filter(data, n_profiles == x, Model == "PGRNN") %>% pull(`Train RMSE`) %>% mean
    pg_mean$train <- c(pg_mean$train, this_train)
    pg_mean$y <- c(pg_mean$y, this_mean)
    pg_mean$x <- c(pg_mean$x, x+pgrnn_offsets[1])
    lines(c(x+pgrnn_offsets[1], x+pgrnn_offsets[1]), c(filter(data, n_profiles == x, Model == "PGRNN") %>% pull(`Test RMSE`) %>% max(),
                                                       filter(data, n_profiles == x, Model == "PGRNN") %>% pull(`Test RMSE`) %>% min()), col = '#7570b3', lwd = 2.5)

    this_mean <- filter(data, n_profiles == x, Model == "PGRNN_trainonly") %>% pull(`Test RMSE`) %>% mean
    this_train <- filter(data, n_profiles == x, Model == "PGRNN_trainonly") %>% pull(`Train RMSE`) %>% mean
    pgto_mean$train <- c(pgto_mean$train, this_train)
    pgto_mean$y <- c(pgto_mean$y, this_mean)
    pgto_mean$x <- c(pgto_mean$x, x)
    lines(c(x, x), c(filter(data, n_profiles == x, Model == "PGRNN_trainonly") %>% pull(`Test RMSE`) %>% max(),
                                                       filter(data, n_profiles == x, Model == "PGRNN_trainonly") %>% pull(`Test RMSE`) %>% min()), col = '#FF007f', lwd = 2.5)

    rnn_offsets <- tail(rnn_offsets, -1L)
    pgrnn_offsets <- tail(pgrnn_offsets, -1L)
  }

  lines(glm_mean$x, glm_mean$y, col = '#1b9e77', lty = 'dashed')
  points(glm_mean$x, glm_mean$y, col = '#1b9e77', pch = 21, bg = 'white', lwd = 2.5, cex = 1.5)
  lines(rnn_mean$x, rnn_mean$y, col = '#d95f02', lty = 'dashed')
  points(rnn_mean$x, rnn_mean$y, col = '#d95f02', pch = 22, bg = 'white', lwd = 2.5, cex = 1.5)
  lines(pg_mean$x, pg_mean$y, col = '#7570b3', lty = 'dashed')
  points(pg_mean$x, pg_mean$y, col = '#7570b3', pch = 23, bg = 'white', lwd = 2.5, cex = 1.5)
  lines(pgto_mean$x, pgto_mean$y, col = '#FF007f', lty = 'dashed')
  points(pgto_mean$x, pgto_mean$y, col = '#FF007f', pch = 23, bg = 'white', lwd = 2.5, cex = 1.5, lty = 'dashed')
  points(glm_mean$x[1], glm_mean$y[1], col = '#1b9e77', pch = 8, lwd = 2.5, cex = 0.6)

  message('PRGNN:', tail(pg_mean$y,1), "\nRNN:", tail(rnn_mean$y,1),'\nGLM:', tail(glm_mean$y, 1))
  message('PRGNN:', tail(pg_mean$train,1), "\nRNN:", tail(rnn_mean$train,1),'\nGLM:', tail(glm_mean$train, 1))

  points(2.2, 0.79, col = '#7570b3', pch = 23, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 0.80, 'Process-Guided Deep Learning', pos = 4, cex = 1.1)

  points(2.2, 0.94, col = '#d95f02', pch = 22, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 0.95, 'Deep Learning', pos = 4, cex = 1.1)

  points(2.2, 1.09, col = '#1b9e77', pch = 21, bg = 'white', lwd = 2.5, cex = 1.5)
  text(2.3, 1.1, 'Process-Based', pos = 4, cex = 1.1)

  dev.off()

}
