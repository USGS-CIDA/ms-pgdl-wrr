## Exploration of data from Jordan that were shared in
## https://drive.google.com/drive/u/0/folders/1MjNrYJ8k4WKKrGAwcaf8ymymYocwQyAD

library(feather)
library(tidyverse)

# read data that Jordan 
geometry <- readr::read_csv('fig_2/in/jordan/mendota_geometry.csv')
meteo <- feather::read_feather('fig_2/in/jordan/mendota_meteo.feather')
temperatures <- feather::read_feather('fig_2/in/jordan/Mendota_temperatures.feather')
training <- feather::read_feather('fig_2/in/jordan/mendota_season_training_500_profiles_experiment_01.feather')
test <- feather::read_feather('fig_2/in/jordan/mendota_season_test.feather')

dim(geometry) # 30 2
head(geometry) # cols: depths, areas

dim(meteo) # 3186 8
head(meteo) # cols: time, ShortWave, LongWave, AirTemp, RelHum, WindSpeed, Rain, Snow

dim(temperatures) # 3185 52
head(temperatures) # cols: DateTime, temp_0, temp_0.5, temp_1, temp_1.5, ..., temp_24.5, ice

dim(training) # 11587 3
head(training) # cols: DateTime, Depth, temp

dim(test) # 15833 3
head(test) # cols: DateTime, Depth, temp
