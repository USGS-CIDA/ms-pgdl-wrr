# Running experiments for manuscript Figure 4 (fig_3)

To use the code files in this directory, start by opening a terminal with working directory set to `ms-pgdl-wrr` (the
grandparent directory of this README.md file).

## Configure a python environment

Install Anaconda Distribution for Python 3.7 (https://www.anaconda.com/distribution/) if needed. Build and
activate the saved Anaconda environment from fig_1/env_pgdl_b.yml<sup>1</sup> with these commands:
```shell script
## shell, working directory = ms-pgdl-wrr ##
conda env create -f fig_3/env_pgdl_b.yml -n pgdl_b
conda activate pgdl_b
```

After these commands, we recommend starting up python in a second window so that variables created in the following
code snippets can persist between snippets.
```shell script
## NEW shell, same working directory ##
conda activate pgdl_b
python
```

## Prepare directories

Create local, temporary directories to hold model inputs and outputs.

```python
## python ##
import os
lake_name = 'nhd_13393567'
all_lakes_path = 'fig_3/tmp/shared'
lake_inputs_path = 'fig_3/tmp/%s/inputs' % lake_name
lake_model_path = 'fig_3/tmp/%s/model' % lake_name
lake_predictions_path = 'fig_3/tmp/%s/out' % lake_name
if not os.path.isdir(all_lakes_path): os.makedirs(all_lakes_path)

if not os.path.isdir(lake_inputs_path): os.makedirs(lake_inputs_path)

if not os.path.isdir(lake_model_path): os.makedirs(lake_model_path)

if not os.path.isdir(lake_predictions_path): os.makedirs(lake_predictions_path)

```

## Prepare model inputs

### Download data from ScienceBase

```python
## python ##
import re
from zipfile import ZipFile
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
met_zip = download_from_sciencebase('5d98e0a3e4b0c4f70d1186ee', '68_lakes_meteo.zip', all_lakes_path)
ice_zip = download_from_sciencebase('5d98e0a3e4b0c4f70d1186ee', '68_pretrainer_ice_flags.zip', all_lakes_path)
glm_zip = download_from_sciencebase('5d915c8ee4b0c4f70d0ce520', '%s_predict.zip' % lake_name, lake_inputs_path)
train_obs_file = download_from_sciencebase('5d8a47bce4b0c4f70d0ae61f', 'all_lakes_historical_training.csv', all_lakes_path)
test_obs_file = download_from_sciencebase('5d925048e4b0c4f70d0d0596', 'all_test.csv', all_lakes_path)

# Unzip
met_file = ZipFile(met_zip, 'r').extract('%s_meteo.csv' % lake_name, lake_inputs_path)
ice_file = ZipFile(ice_zip, 'r').extract('%s_ice_flag.csv' % lake_name, lake_inputs_path)
glm_file = ZipFile(glm_zip, 'r').extract('%s_predict_pb0.csv' % lake_name, lake_inputs_path)
```

### Munge data for pretraining, training, and testing

Add lake-specific training and test data to the lake-specific inputs folder.
```python
## python ##
import pandas as pd

# define the filenames again if already downloaded from ScienceBase in a previous python session
train_obs_file = os.path.join(all_lakes_path, 'all_lakes_historical_training.csv')
test_obs_file = os.path.join(all_lakes_path, 'all_test.csv')

# read, subset, and write the training data for a single experiment
train_obs = pd.read_csv(train_obs_file)
train_obs_subset = train_obs[(train_obs['site_id'] == lake_name)].reset_index()[['date','depth','temp']]
train_obs_subset.to_feather(os.path.join(lake_inputs_path, 'labels_train.feather'))

# read, subset, and write the testing data for a single experiment
test_obs = pd.read_csv(test_obs_file)
test_obs_subset = test_obs[(train_obs['site_id'] == lake_name)].reset_index()[['date','depth','temp']]
test_obs_subset.to_feather(os.path.join(lake_inputs_path, 'labels_test.feather'))
```

Next generate prepared .npy files in the inputs folder.
Use `fig_3/src/preprocess_pretrain_data.py` and `fig_3/src/preprocess_train_test_data.py`
for this step.

## Train and predict with PGDL

Run `pgrnn_figure3.py` to pretrain, train, and generate predictions 5 times for the lake. That file also uses functions
from the following module files:
```text
io_operations.py
phys_operations.py
preprocess_functions.py
pytorch_data_operations.py
pytorch_model_operations.py
```

Output includes model checkpoints after pretraining and then after training, plus a file of predictions for each
of the 5 replicates per model.

## Exit

Now that training is complete, if you plan to use either of the open shells for other operations,
you may want to deactivate the `pgdl_b` conda environment before proceeding:
```shell script
## shell ##
conda deactivate
```


## Footnote

<sup>1</sup>We created the Anaconda environment with these commands:
```shell script
## shell; no need to run these lines ##
conda create -n pgdl_b python=3.6.8
conda install -n pgdl_b pytorch=0.4.1 -c pytorch
conda install -n pgdl_b pandas=0.23.4 scikit-learn=0.20.1 requests=2.18.4
conda install -n pgdl_b feather-format=0.4.0 -c conda-forge
conda activate pgdl_b
pip install sciencebasepy
conda deactivate
conda env export -n pgdl_b | grep -v "^prefix: " > fig_3/env_pgdl_b.yml
```
