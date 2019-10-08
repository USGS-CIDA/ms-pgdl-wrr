# Code files for running Experiment 1 (Fig 1)

To use the code files in this directory, start by opening a terminal with working directory set to `ms-pgdl-wrr` (the
grandparent directory of this README.md file).

## Downloading PGDL model inputs from ScienceBase

### Download data from ScienceBase

```python
import os
import sciencebasepy
import re

# Configure access to ScienceBase access
sb = sciencebasepy.SbSession()
# Th following line should only be necessary before the data release is public:
sb.login('[username]', '[password]') # manually revise username and password

raw_data_path = 'fig_1/tmp/mendota/pretrain/inputs/raw'
os.makedirs(raw_data_path, exist_ok=True)

def download_from_sciencebase(item_id, search_text, to_folder):
    item_info = sb.get_item(item_id)
    file_info = [f for f in item_info['files'] if re.search(search_text, f['name'])][0]
    sb.download_file(file_info['downloadUri'], file_info['name'], to_folder)
    return os.path.join(to_folder, file_info['name'])
met_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'meteo.csv', raw_data_path)
glm_file = download_from_sciencebase('5d915cb2e4b0c4f70d0ce523', 'predict_pb0.csv', raw_data_path)
ice_file = download_from_sciencebase('5d98e0c4e4b0c4f70d1186f1', 'pretrainer_ice_flags.csv', raw_data_path)
```

### All models: lake hypsography, prediction features, GLM predictions, and ice cover flag

```shell script
python fig_1/src/processing_USGS.py --lake_name mendota \
  --met_file fig_1/tmp/mendota/pretrain/inputs/raw/mendota_meteo.csv \
  --glm_file fig_1/tmp/mendota/pretrain/inputs/raw/me_predict_pb0.csv \
  --ice_file fig_1/tmp/mendota/pretrain/inputs/raw/mendota_pretrainer_ice_flags.csv \
  --processed_path fig_1/tmp/mendota/pretrain/inputs/ready
```
where `lake_name` can be `mendota` or `sparkling`.



### Pretraining data: Process-based model predictions

### Training data: Temperature observations



## Preparing data files for PGDL training

```shell script
chmod 777 fig_1/src/processing_USGS.py
python fig_1/src/processing_USGS.py
```

## Training and predicting with PGDL

We created an Anaconda environment and saved it with this command:
```shell script
conda env export > environment.yml
```

You can now recreate and load that environment with these commands:
```shell script
conda env create -f environment.yml
conda activate tf_cpu
```

It may be necessary to provide Execute permissions for the python scripts.
```shell script
chmod 777 fig_1/src/PGRNN_pretrain_USGS.py
chmod 777 fig_1/src/PGRNN_USGS.py
```

### Pretrain

```shell script
python fig_1/src/PGRNN_pretrain_USGS.py --data_path fig_1/tmp/mendota/pretrain/inputs/ready --save_path fig_1/tmp/mendota/pretrain/model
```

### Train

```shell script
python fig_1/src/PGRNN_USGS.py --data_path fig_1/tmp/mendota/train/inputs/ready --restore_path fig_1/tmp/mendota/pretrain/model --save_path fig_1/tmp/mendota/train/model
```
where `restore_path` in this training command should equal `save_path` from the pretraining command.