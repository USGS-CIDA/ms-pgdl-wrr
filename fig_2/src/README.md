# Code files for running Experiment 2 (Fig 2)

## Example set of files for a single model run (pretraining or training) for Mendota or Sparkling

These files were emailed by Xiaowei on 1/30/19

* processing_mendota.py, processing_sparkling.py - preparing the data. The resulting column names are available from Jordan's shared data in Google Drive. The rows are different depths with the interval of 0.5m.
* PGRNN_EC_pretrain_mendota.py, PGRNN_EC_pretrain_sparkling.py - model pretraining (with 'observations' from X_temperatures.feather). For the pretraining, I attached the code "pretraining_mendota/sparkling.py". These scripts are exactly the same model as in PGRNN_ECp_season_exp1_mendota.py etc., but just use simulated data to pre-train and then save the model for fine-tuning.
* PGRNN_ECp_season_exp1_mendota.py, PGRNN_ECp_season_exp1_sparkling.py - model training (with observations from X_Y_training_Z_profiles_experiment_01.feather)

Xiaowei tried a few parameter settings for Sparkling Lake, but it seems that it can get no better than using the same parameters as for Mendota, so they're the same for the two lakes.
