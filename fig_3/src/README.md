# Code for Figure 3
Input data prepared by Jordan Read
Python scripts by Jared Willard
R script by Alison Appling

Figure 3: Improvements in water temperatures predictions (as estimated by Root Mean Squared Error; RMSE) for lakes between the pre-trainer (uncalibrated process-based predictions of temperature used to initialize the PGDL network) and the PGDL (Process-Guided Deep Learning as described in Figure 1). The PGDL modeling process was automated and scaled up to simulate temperatures for 68 lakes in the midwest U.S. Green dots aligned with pre-trainer are the RMSEs of the General Lake Model (Hipsey et al. 2019) temperature predictions after lake-specific parameterization, but with no calibration to water temperature observations. Purple diamonds are the RMSE of PGDL models that used pre-trainer as supervised training data to initialize node weights and network structure before a second (and final) training with in-situ water temperature observations. Lines connecting lake-specific pre-trainer RMSEs to paired PGDL RMSEs are colored according to the magnitude of improvement between the two models, where yellow lines represent marginal improvement and dark grey colors represent large improvements. Density plots for pre-trainer and PGDL RMSEs are shown in the right side margin of the plot.

# File descriptions

Jared's scripts:
* takeme.sh is just shorthand for setting up the conda environment on my MSI account with all the necessary libraries for pytorch/numpy/pandas and also takes you to the current working directory. 
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/data/preprocess_manylakes.py - used for preprocessing
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/data/rw_data.py - obsolete and no longer used
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/data/pytorch_data_operations.py - helper functions and data setup mostly
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/models/pytorch_model_operations.py - helper functions and data setup mostly
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/scripts/manylakes/experiment_correlation_check.py - this is the most recent version, and experiment_correlation_check_small_batch.py was used for just one lake (13393533) that needed a smaller batch size because it had less data.
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/scripts/manylakes/jobcreate.py - creates the set of job scripts and writes qsub_script.sh
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/scripts/manylakes/job_2385496.sh and similar - lake-specific job scripts
* https://github.com/jdwillard19/lake_modeling/blob/jared/src/scripts/manylakes/qsub_script.sh - script to run all lake-specific job scripts on cluster
