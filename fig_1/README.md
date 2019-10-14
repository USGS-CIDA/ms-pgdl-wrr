# Running experiments for Figures 2 and 3

To use the code files in this directory, start by opening a terminal with working directory set to `ms-pgdl-wrr` (the
grandparent directory of this README.md file).

## Configure a python environment

Install Anaconda Distribution for Python 2.7 (https://www.anaconda.com/distribution/) if needed.

We created an Anaconda environment and saved it with this command:
```shell script
## shell ##
conda env export > environment.yml
```

You can now recreate and load that environment with these commands:
```shell script
## shell ##
conda env create -f environment.yml
conda activate tf_cpu
```

## Prepare directories

Create local, temporary directories to hold model inputs and outputs.

```python
## python ##
import os
raw_data_path = 'fig_1/tmp/mendota/shared/raw_data'
pretrain_inputs_path = 'fig_1/tmp/mendota/pretrain/inputs'
train_inputs_path = 'fig_1/tmp/mendota/train/inputs'
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(pretrain_inputs_path, exist_ok=True)
os.makedirs(train_inputs_path, exist_ok=True)
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
sb.login('[username]', '[password]') # manually revise username and password

def download_from_sciencebase(item_id, search_text, to_folder):
    item_info = sb.get_item(item_id)
    file_info = [f for f in item_info['files'] if re.search(search_text, f['name'])][0]
    sb.download_file(file_info['downloadUri'], file_info['name'], to_folder)
    return os.path.join(to_folder, file_info['name'])
# URLs can be browsed by adding one of the following IDs after https://www.sciencebase.gov/catalog/item/,
# e.g., https://www.sciencebase.gov/catalog/item/5d98e0c4e4b0c4f70d1186f1
met_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'meteo.csv', raw_data_path)
ice_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'pretrainer_ice_flags.csv', raw_data_path)
glm_file = download_from_sciencebase('5d915cb2e4b0c4f70d0ce523', 'predict_pb0.csv', raw_data_path)
train_obs_file = download_from_sciencebase('5d8a837fe4b0c4f70d0ae8ac', 'similar_training.csv', raw_data_path)
test_obs_file = download_from_sciencebase('5d925066e4b0c4f70d0d0599', 'test.csv', raw_data_path)
```

### Munge data for pretraining, training, and testing

First generate prepared .npy files in the pretraining inputs folder. `lake_name` can be `mendota` or `sparkling`.
The processing_USGS.py script also generates GLM predictions file labels_pretrain.npy, which are only used (1) to set
the date range of the meteorological inputs and (2) as "observations" for pretraining.
```shell script
## shell ##
python fig_1/src/processing_USGS.py \
  --phase pretrain \
  --lake_name mendota \
  --met_file fig_1/tmp/mendota/shared/raw_data/mendota_meteo.csv \
  --glm_file fig_1/tmp/mendota/shared/raw_data/me_predict_pb0.csv \
  --ice_file fig_1/tmp/mendota/shared/raw_data/mendota_pretrainer_ice_flags.csv \
  --processed_path fig_1/tmp/mendota/pretrain/inputs
```

Do the same processing for training, deleting labels_pretrain.npy because it's not needed after this processing step.
Note that the different `phase` argument causes a different subset of data to be exported.
```shell script
## shell ##
python fig_1/src/processing_USGS.py \
  --phase train \
  --lake_name mendota \
  --met_file fig_1/tmp/mendota/shared/raw_data/mendota_meteo.csv \
  --glm_file fig_1/tmp/mendota/shared/raw_data/me_predict_pb0.csv \
  --ice_file fig_1/tmp/mendota/shared/raw_data/mendota_pretrainer_ice_flags.csv \
  --processed_path fig_1/tmp/mendota/train/inputs
rm fig_1/tmp/mendota/train/inputs/labels_pretrain.npy
```

Add training and test data to the training inputs folder.
```python
## python ##
import pandas as pd

# read, subset, and write the training data for a single experiment
train_obs = pd.read_csv(train_obs_file)
train_obs_similar_980_1 = train_obs[(train_obs['exper_id'] == 'similar_980') & (train_obs['exper_n'] == 1)].reset_index()[['date','depth','temp']]
train_obs_similar_980_1.to_feather(os.path.join(train_inputs_path, 'labels_train.feather'))

# read, subset, and write the testing data for a single experiment
test_obs = pd.read_csv(test_obs_file)
test_obs_similar_1 = test_obs[(test_obs['exper_type'] == 'similar') & (test_obs['exper_n'] == 1)].reset_index()[['date','depth','temp']]
test_obs_similar_1.to_feather(os.path.join(train_inputs_path, 'labels_test.feather'))
```

## Train and predict with PGDL

### Pretrain

```shell script
## shell ##
python fig_1/src/PGRNN_pretrain_USGS.py \
  --data_path fig_1/tmp/mendota/pretrain/inputs \
  --save_path fig_1/tmp/mendota/pretrain/model
```

### Train

```shell script
## shell ##
python fig_1/src/PGRNN_USGS.py \
  --data_path fig_1/tmp/mendota/train/inputs \
  --restore_path fig_1/tmp/mendota/pretrain/model \
  --save_path fig_1/tmp/mendota/train/model
```
where `restore_path` in this training command equals `save_path` from the pretraining command.