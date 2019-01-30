# Input files from Jordan

Data files are stored in https://drive.google.com/drive/u/0/folders/1MjNrYJ8k4WKKrGAwcaf8ymymYocwQyAD

## Example set of files for a single model run (pretraining + training):

* mendota_geometry.csv - lake shape; cols for depths, areas
* mendota_meteo.feather - daily values for ShortWave, LongWave, AirTemp, RelHum, WindSpeed, Rain, Snow
* Mendota_temperatures.feather - GLM-predicted temperatures, for pre-training, in wide format (1 row per date, 1 col per depth)
* mendota_season_training_500_profiles_experiment_01.feather - fine-tuning observations, cols for DateTime, Depth, temp
* mendota_season_test.feather - test observations, cols for DateTime, Depth, temp

# Other files in this folder

* mendota_season_overview.png - graphic of training and test data
* explore_jordan_data.R - script to inspect the data in R
