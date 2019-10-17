library(dplyr)
library(ggplot2)
############################################
# figure of number of lakes vs days of obs
############################################
# Fig. S2

# read in all obs temp data
# file downloaded from: https://drive.google.com/drive/u/1/folders/1pbhIjfYUPZ4lEICm5zwJFjIjGYEz1qwi

all_dat <- feather::read_feather('~/Downloads/merged_temp_data_daily.feather')

counts <- all_dat %>%
  group_by(nhd_id, date) %>%
  summarize(depths = n()) %>%
  filter(depths >= 5) %>% # this is consistent with how 68 lakes were chosen
  group_by(nhd_id) %>%
  summarize(lakedays = n())

numlakes <- function(lakedays, benchmark) {
  nlakes <- c()
  for (i in 1:length(benchmark)) {
    nlakes[i] <- length(which(lakedays >= benchmark[i]))
  }
  return(nlakes)
}

vals <- numlakes(counts$lakedays, benchmark = seq(1, 1500, 1))

plot_dat <- data.frame(lakedays = seq(1, 1500, 1), nlakes = vals)

vsegments <- filter(plot_dat, lakedays %in% c(2, 10, 50, 100, 500, 980))

p <- ggplot(plot_dat, aes(x = lakedays, y = nlakes)) +
  geom_line() +
  theme_classic() +
  scale_x_log10(breaks = c(2, 10, 50, 100, 500, 980)) +
  labs(x = 'Number of unique observation dates',
       y = 'Lakes with temperature observations') +
  geom_point(data = vsegments, aes(x = lakedays, y = nlakes),
             shape = 21, fill = 'white') +
  geom_text(data = vsegments,
            aes(x = lakedays, y = nlakes,
                label = c('3602 lakes', '1736 lakes', '558 lakes', '267 lakes', '9 lakes', '2 lakes')),
            hjust = c(rep(0, 4), 0.5, 0.3),
            vjust = c(rep(0,4), -0.7, -0.6),
            nudge_x = 0.05, size = 3)

ggsave(filename = 'figures/supp_fig_S2.png', height = 4, width = 4)

