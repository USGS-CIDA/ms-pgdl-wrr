# Code files for running Experiment 1 (Fig 1)

## Training and predicting with PGDL

```shell script
conda activate tf_cpu
conda env export > environment.yml
```

```shell script
conda env create -f environment.yml
conda activate tf_cpu
chmod 777 fig_1/src/PGRNN_USGS.py
python fig_1/src/PGRNN_USGS.py --data_path fig_1/tmp/mendota/train/inputs/ready --restore_path fig_1/tmp/mendota/pretrain/model --save_path fig_1/tmp/train/model
```


## Example scripts for a single scenario

* processing_mendota.py - preparing the data. The resulting column names are available from Jordan's shared data in Google Drive. The rows are different depths with the interval of 0.5m.
* PGRNN_sliding_ECp_exp4_980.py - model training (pretraining script is nearly identical; see scripts for Fig 2)
