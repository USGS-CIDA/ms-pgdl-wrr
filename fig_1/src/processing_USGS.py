# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:45:53 2018

@author: jiaxx
"""

# if needed, install sciencebasepy:
# pip install sciencebasepy
# conda install requests

import pandas as pd
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=['pretrain', 'train'])
parser.add_argument('--lake_name', choices=['mendota', 'sparkling'])
parser.add_argument('--met_file')
parser.add_argument('--glm_file')
parser.add_argument('--ice_file')
parser.add_argument('--processed_path')
args = parser.parse_args()

# hard code and save some depth-area relationship and other lake-specific tidbits
if args.lake_name == 'mendota':
    depth_areas = np.array([
        39865825,38308175,38308175,35178625,35178625,33403850,31530150,31530150,30154150,30154150,29022000,
        29022000,28063625,28063625,27501875,26744500,26744500,26084050,26084050,25310550,24685650,24685650,
        23789125,23789125,22829450,22829450,21563875,21563875,20081675,18989925,18989925,17240525,17240525,
        15659325,14100275,14100275,12271400,12271400,9962525,9962525,7777250,7777250,5956775,4039800,4039800,
        2560125,2560125,820925,820925,216125])
    data_chunk_size = 5295 # size of half the dates in the training period
elif args.lake_name == 'sparkling':
    depth_areas = np.array([
        637641.569, 637641.569, 592095.7426, 592095.7426, 546549.9163, 546549.9163, 546549.9163, 501004.0899,
        501004.0899, 501004.0899, 455458.2636, 455458.2636, 409912.4372, 409912.4372, 409912.4372, 364366.6109,
        364366.6109, 318820.7845, 318820.7845, 318820.7845, 273274.9581, 273274.9581, 273274.9581, 227729.1318,
        227729.1318, 182183.3054, 182183.3054, 182183.3054, 136637.4791, 136637.4791, 136637.4791, 91091.65271,
        91091.65271, 45545.82636, 45545.82636, 45545.82636, 0])
    data_chunk_size = 5478
n_depths = depth_areas.size
np.save(os.path.join(args.processed_path, 'depth_areas.npy'), depth_areas)
np.save(os.path.join(args.processed_path, 'data_chunk_size.npy'), data_chunk_size)

# Read data files
feat = pd.read_csv(args.met_file)
glm = pd.read_csv(args.glm_file)

# Truncate to the training or testing period
if args.phase == 'pretrain':
    feat = feat[pd.to_datetime(feat['date'].values) <= pd.to_datetime('2009-04-01')]
elif args.phase == 'train':
    feat = feat[pd.to_datetime(feat['date'].values) > pd.to_datetime('2009-04-01')]

# Truncate feat to dates with glm predictions and vice versa
feat = feat.merge(glm[['date']], on='date')
glm = glm.merge(feat[['date']], on='date')

# create dates, x_full, x_raw_full, diag_full, label(glm)
x_raw_full = feat.drop('date', axis=1).values # ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
new_dates = feat[['date']].values[:,0]
np.save(os.path.join(args.processed_path, 'dates.npy'), new_dates)


n_steps = x_raw_full.shape[0]

import datetime
format = "%Y-%m-%d"

doy = np.zeros([n_steps,1])
for i in range(n_steps):
    dt = datetime.datetime.strptime(str(new_dates[i]), format)
    tt = dt.timetuple()
    doy[i,0] = tt.tm_yday
    
  
x_raw_full = np.concatenate([doy,np.zeros([n_steps,1]),x_raw_full],axis=1) # ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
x_raw_full = np.tile(x_raw_full,[n_depths,1,1]) # add depth replicates as prepended first dimension

for i in range(n_depths):
    x_raw_full[i,:,1] = i*0.5 # fill in the depth column as depth in m (0, 0.5, 1, ..., (n_depths-1)/2)

# copy into matrix, still with columns ['DOY', 'depth', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow']
x_raw_full_new = np.zeros([x_raw_full.shape[0],x_raw_full.shape[1],x_raw_full.shape[2]],dtype=np.float64)
for i in range(x_raw_full.shape[0]):
    for j in range(x_raw_full.shape[1]):
        for k in range(x_raw_full.shape[2]):
            x_raw_full_new[i,j,k] = x_raw_full[i,j,k]
            
np.save(os.path.join(args.processed_path, 'features.npy'),x_raw_full_new)
x_raw_full = np.load(os.path.join(args.processed_path, 'features.npy'))

# standardize features
from sklearn import preprocessing
x_full = preprocessing.scale(np.reshape(x_raw_full,[n_depths*n_steps,x_raw_full.shape[-1]]))
x_full = np.reshape(x_full,[n_depths,n_steps,x_full.shape[-1]])
np.save(os.path.join(args.processed_path, 'processed_features.npy'),x_full)


# label_glm 
glm_new = glm.drop('date', axis=1).values
glm_new = np.transpose(glm_new)

labels = np.zeros([n_depths,n_steps],dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        labels[i,j] = glm_new[i,j]

np.save(os.path.join(args.processed_path, 'labels_pretrain.npy'), labels)


# phy files ------------------------------------------------------------
diag_all = pd.read_csv(args.ice_file)
diag_merged = diag_all.merge(feat, how='right', on='date')[['ice']].values

diag = np.zeros([n_depths, n_steps, 3], dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        diag[i,j,2] = diag_merged[j,0]
np.save(os.path.join(args.processed_path, 'diag.npy'),diag) # ['ignored', 'ignored', 'ice']

print("Processed data are in %s" % args.processed_path)
