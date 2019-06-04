## GLM runs on Yeti

We are running three param calibrations on 68 lakes, using `optim()` and writing the final params, test/train RMSE, site_id, and experiment details to file for each lake.

nml files, driver files, and test/train data are built locally using task tables, then the directories are synched to Yeti dirs.

We are using simple 1-68 array batch jobs on Yeti, and one batch per experiment group (total of 3 for sparse simulations, 1 for complete sims)

We run a clean-up/summarize script locally on Yeti when the jobs are done, which combines all of the result file snippets into a single file. 

As of 5/25/2019, on Yeti we need to 
```
module load tools/nco-4.7.8-gnu
module load tools/netcdf-4.3.2-gnu
```
before library installs and as part of the batch file

In the dir that we are running from, we need to point to our local library installs. For me, 

```
R_LIBS=/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/glm-optim-wrr/Rlib
```
(To do: capture install script so we can rebuild these libraries as needed)

The script that runs for each job is simple 
 - we use the job array ID as the index in our job csv/feather table to figure out the site_id and identify the files to grab
 - we copy all sim files and test/train files into a temporary folder named based on the site_id and experiment_id
 - we run the optim on the training data and write results to a file in the results dir
 - we unlink the temp folder that we used for sims
 
When the jobs are complete
 - we run the summarize R script that collects all of the results and writes a single table
 
to set up:
```
cd /cxfs/projects/usgs/water/iidd/data-sci/lake-temp/glm-optim-wrr
```
`out` is where the results go 
`out/fig_3` is where figure 3 results are written to
`in` is where we sync external data to
`in/fig_3` is where we sync figure 3 data to
`Rlib` is where our R packages are installed to
`src` is where our R scripts are
`sim-scratch` is where sims are run (in sub dirs)
`sim-scratch/fig_3/nhd_10596466_010_profiles_experiment_01` is an example simulation directory that is created and then unlinked when complete

`sbatch {array_job_names}.batch` to start the jobs
`squeue -u jread --start` to check job status

to build the files for yeti sync:
```r
scmake('fig_3/out/figure3_data_files.feather','figure_3_data_remake.yml')
```

to sync data to Yeti:
```
cd fig_3/yeti_sync
rsync -avz .  jread@yeti.cr.usgs.gov:/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/glm-optim-wrr/in/fig_3
```

to sync R script to Yeti:
```
cd fig_3/src
rsync yeti_src_glm_optim.R jread@yeti.cr.usgs.gov:/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/glm-optim-wrr/src/yeti_src_glm_optim.R
```

to sync job table to Yeti:
```
cd fig_3/out
rsync fig_3_yeti_jobs.csv jread@yeti.cr.usgs.gov:/cxfs/projects/usgs/water/iidd/data-sci/lake-temp/glm-optim-wrr/in/fig_3_yeti_jobs.csv
```

on yeti, can tail a log w/ 
```
tail sim-scratch/{nhd_10596466_010_profiles_experiment_01}/results_log.txt
```

see what jobs are done and/or count complete jobs with:
```
ls out/fig_3

ls out/fig_3/ | wc -l
```
