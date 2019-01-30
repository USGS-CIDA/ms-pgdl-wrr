## Exploration of data from Jordan that were shared in
## https://drive.google.com/drive/u/0/folders/1MjNrYJ8k4WKKrGAwcaf8ymymYocwQyAD

library(feather)
library(tidyverse)

# read data that Jordan 
geometry <- readr::read_csv('fig_2/in/jordan/mendota_geometry.csv')
meteo <- feather::read_feather('fig_2/in/jordan/mendota_meteo.feather')
temperatures <- feather::read_feather('fig_2/in/jordan/Mendota_temperatures.feather') # GLM-predicted temperatures
training <- feather::read_feather('fig_2/in/jordan/mendota_season_training_500_profiles_experiment_01.feather')
test <- feather::read_feather('fig_2/in/jordan/mendota_season_test.feather')

head(temperatures) 
head(training)
