# Running experiments for manuscript Figure 3 (fig_2)

The experiments for Figure 3 use the same python code as those for Figure 2, just with different inputs.
For this README example we will demonstrate a model run for Sparkling Lake,

To use the code files relevant to this figure, start by opening a terminal with working directory set to `ms-pgdl-wrr` (the
grandparent directory of this README.md file).

## Configure a python environment

Install Anaconda Distribution for Python 2.7 (https://www.anaconda.com/distribution/) if needed. Build and
activate the saved Anaconda environment from fig_1/env_pgdl_a.yml<sup>1</sup> with these commands:
```shell script
## shell, working directory = ms-pgdl-wrr ##
conda env create -f fig_1/env_pgdl_a.yml -n pgdl_a
conda activate pgdl_a
```

After these commands, we recommend starting up python in a second window so that variables created in the following
code snippets can persist between snippets.
```shell script
## NEW shell, same working directory ##
conda activate pgdl_a
python
```

## Prepare directories

Create local, temporary directories to hold model inputs and outputs.

```python
## python ##
import os
raw_data_path = 'fig_1/tmp/sparkling/shared/raw_data'
pretrain_inputs_path = 'fig_1/tmp/sparkling/pretrain/inputs'
pretrain_model_path = 'fig_1/tmp/sparkling/pretrain/model'
train_inputs_path = 'fig_2/tmp/sparkling/train/year_1/inputs'
train_model_path = 'fig_2/tmp/sparkling/train/year_1/model'
predictions_path = 'fig_2/tmp/sparkling/train/year_1/out'
if not os.path.isdir(raw_data_path): os.makedirs(raw_data_path)

if not os.path.isdir(pretrain_inputs_path): os.makedirs(pretrain_inputs_path)

if not os.path.isdir(pretrain_model_path): os.makedirs(pretrain_model_path)

if not os.path.isdir(train_inputs_path): os.makedirs(train_inputs_path)

if not os.path.isdir(train_model_path): os.makedirs(train_model_path)

if not os.path.isdir(predictions_path): os.makedirs(predictions_path)

```

## Prepare model inputs

### Download data from ScienceBase

```python
## python ##
import re
import sciencebasepy
# Configure access to ScienceBase access
sb = sciencebasepy.SbSession()
# Th following line should only be necessary before the data release is public:
# sb.login('[username]', '[password]') # manually revise username and password

def download_from_sciencebase(item_id, search_text, to_folder):
    item_info = sb.get_item(item_id)
    file_info = [f for f in item_info['files'] if re.search(search_text, f['name'])][0]
    sb.download_file(file_info['downloadUri'], file_info['name'], to_folder)
    return os.path.join(to_folder, file_info['name'])

# Data release URLs can be browsed by adding one of the following IDs after "https://www.sciencebase.gov/catalog/item/",
# e.g., https://www.sciencebase.gov/catalog/item/5d98e0c4e4b0c4f70d1186f1
met_file = download_from_sciencebase('5d98e0dbe4b0c4f70d1186f3', 'meteo.csv', raw_data_path)
ice_file = download_from_sciencebase('5d98e0dbe4b0c4f70d1186f3', 'pretrainer_ice_flags.csv', raw_data_path)
glm_file = download_from_sciencebase('5d915cc6e4b0c4f70d0ce525', 'predict_pb0.csv', raw_data_path)
train_obs_file = download_from_sciencebase('5d8a4752e4b0c4f70d0ae61a', 'year_training.csv', raw_data_path)
test_obs_file = download_from_sciencebase('5d92507be4b0c4f70d0d059b', 'test.csv', raw_data_path)
```

### Munge data for pretraining, training, and testing

First generate prepared .npy files in the pretraining inputs folder. `lake_name` can be `mendota` or `sparkling`.
The processing_USGS.py script also generates GLM predictions file labels_pretrain.npy, which are only used (1) to set
the date range of the meteorological inputs and (2) as "observations" for pretraining. Note that if this has already
been done for fig_1, there is no need to do it again for fig_2.
```shell script
## shell ##
python fig_1/src/processing_USGS.py \
  --phase pretrain \
  --lake_name sparkling \
  --met_file fig_1/tmp/sparkling/shared/raw_data/sparkling_meteo.csv \
  --glm_file fig_1/tmp/sparkling/shared/raw_data/sp_predict_pb0.csv \
  --ice_file fig_1/tmp/sparkling/shared/raw_data/sparkling_pretrainer_ice_flags.csv \
  --processed_path fig_1/tmp/sparkling/pretrain/inputs
```

Do the same processing for training, deleting labels_pretrain.npy because it's not needed for training.
Note that the different `phase` argument causes a different subset of data to be exported. Because
training is different for each model, save these outputs to a new model-specific folder within fig_2/.
```shell script
## shell ##
python fig_1/src/processing_USGS.py \
  --phase train \
  --lake_name sparkling \
  --met_file fig_1/tmp/sparkling/shared/raw_data/sparkling_meteo.csv \
  --glm_file fig_1/tmp/sparkling/shared/raw_data/sp_predict_pb0.csv \
  --ice_file fig_1/tmp/sparkling/shared/raw_data/sparkling_pretrainer_ice_flags.csv \
  --processed_path fig_2/tmp/sparkling/train/year_1/inputs
rm fig_2/tmp/sparkling/train/year_1/inputs/labels_pretrain.npy
```

Add training and test data to the training inputs folder. labels_train.feather and labels_test.feather are the only
files that differ from model to model (for the same lake) for the experiment represented by this figure.
```python
## python ##
import pandas as pd

# define the filenames again if already downloaded from ScienceBase in a previous python session
train_obs_file=os.path.join(raw_data_path, 'sp_year_training.csv')
test_obs_file=os.path.join(raw_data_path, 'sp_test.csv')

# read, subset, and write the training data for a single experiment
train_obs = pd.read_csv(train_obs_file)
train_obs_subset = train_obs[(train_obs['exper_id'] == 'year_500') & (train_obs['exper_n'] == 1) & (~np.isnan(train_obs['temp']))].reset_index()[['date','depth','temp']]
train_obs_subset.to_feather(os.path.join(train_inputs_path, 'labels_train.feather'))

# read, subset, and write the testing data for a single experiment
test_obs = pd.read_csv(test_obs_file)
test_obs_subset = test_obs[(test_obs['exper_type'] == 'year') & (test_obs['exper_n'] == 1) & (~np.isnan(test_obs['temp']))].reset_index()[['date','depth','temp']]
test_obs_subset.to_feather(os.path.join(train_inputs_path, 'labels_test.feather'))
```

## Train and predict with PGDL

The following commands execute the pretraining and training phases of preparing a PGDL model. 
There are some deprecation warnings that you can safely ignore.

### Pretrain

Pretraining only needs to be done once per lake for both fig_1 and fig_2.

```shell script
## shell ##
export KMP_DUPLICATE_LIB_OK=TRUE
python fig_1/src/PGRNN_pretrain_USGS.py \
  --data_path fig_1/tmp/sparkling/pretrain/inputs \
  --save_path fig_1/tmp/sparkling/pretrain/model
```

### Train

In addition to updating PGDL weights and parameters based on training observations, the following training script
includes some code to generate predictions and compare them to test data. The test data are only used for this purpose.

```shell script
## shell ##
python fig_1/src/PGRNN_USGS.py \
  --data_path fig_2/tmp/sparkling/train/year_1/inputs \
  --restore_path fig_1/tmp/sparkling/pretrain/model \
  --save_path fig_2/tmp/sparkling/train/year_1/model \
  --preds_path fig_2/tmp/sparkling/train/year_1/out
```
where `restore_path` in this training command equals `save_path` from the pretraining command.


## Exit

Now that training is complete, if you plan to use either of the open shells for other operations,
you may want to deactivate the `pgdl_a` conda environment before proceeding:
```shell script
## shell ##
conda deactivate
```


## Footnote

<sup>1</sup>We created the Anaconda environment with these commands:
```shell script
## shell; no need to run these lines ##
conda create -n pgdl_a python=2.7.15 
conda install -n pgdl_a tensorflow=1.14.0 # to use GPUs use tensorflow-gpu=1.12.0
conda install -n pgdl_a pandas=0.22.0 requests=2.18.4 scikit-learn=0.20.1
conda install -n pgdl_a -c conda-forge feather-format=0.4.0
conda activate pgdl_a
pip install sciencebasepy
conda deactivate
conda env export -n pgdl_a | grep -v "^prefix: " > fig_1/env_pgdl_a.yml
```
