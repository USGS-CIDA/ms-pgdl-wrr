# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:45:53 2018

@author: jiaxx
"""

import pandas as pd
import numpy as np

feat = pd.read_feather('../mendota_meteo.feather')
glm = pd.read_feather('../Generic_GLM_Mendota_temperatures.feather')

feat.columns
feat.values
feat.values.shape


# create x_full, x_raw_full, diag_full, label(glm) 
x_raw_full = feat.values[1:,1:]
new_dates = feat.values[1:,0]
np.save('dates.npy',new_dates)


n_steps = x_raw_full.shape[0]

import datetime
format = "%Y-%m-%d %H:%M:%S"

doy = np.zeros([n_steps,1])
for i in range(n_steps):
    dt = datetime.datetime.strptime(str(new_dates[i]), format)
    tt = dt.timetuple()
    doy[i,0] = tt.tm_yday
    
  
n_depths = 50    
x_raw_full = np.concatenate([doy,np.zeros([n_steps,1]),x_raw_full],axis=1)
x_raw_full = np.tile(x_raw_full,[n_depths,1,1])

for i in range(n_depths):
    x_raw_full[i,:,1] = i*0.5

x_raw_full_new = np.zeros([x_raw_full.shape[0],x_raw_full.shape[1],x_raw_full.shape[2]],dtype=np.float64)
for i in range(x_raw_full.shape[0]):
    for j in range(x_raw_full.shape[1]):
        for k in range(x_raw_full.shape[2]):
            x_raw_full_new[i,j,k] = x_raw_full[i,j,k]
            
np.save('features.npy',x_raw_full_new)
x_raw_full = np.load('features.npy')

# standardize features
from sklearn import preprocessing
x_full = preprocessing.scale(np.reshape(x_raw_full,[n_depths*n_steps,x_raw_full.shape[-1]]))
x_full = np.reshape(x_full,[n_depths,n_steps,x_full.shape[-1]])
np.save('processed_features.npy',x_full)


# label_glm 
glm_new = glm.values[:,1:]
glm_new = np.transpose(glm_new)

labels = np.zeros([n_depths,n_steps],dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        labels[i,j] = glm_new[i,j]

np.save('labels.npy',labels)


# phy files ------------------------------------------------------------
diag_all = pd.read_feather('../Generic_GLM_Mendota_diagnostics.feather')
diag_all.columns

idx = [-11,-10,3]
diag_sel = diag_all.values[:,idx]
diag_sel[:,2] = diag_sel[:,2]>0
diag_sel = np.tile(diag_sel,[n_depths,1,1])


diag = np.zeros([n_depths, n_steps, 3], dtype=np.float64)

for i in range(n_depths):
    for j in range(n_steps):
        diag[i,j,:] = diag_sel[i,j,:]
np.save('diag.npy',diag)


#
#
## debugging -----------------------
#
#import matplotlib.pyplot as plt  
#olen= 1800
#d_sel = 20
#x = range(olen)
##y1 = glm.values[:olen,d_sel+1]
#y1 = labels[d_sel,:olen,]
##y2 = label_o[d_sel,10592:10592+olen]
#plt.plot(x,y1)
##plt.plot(x,y2)
#
#
#f_sel = 8
#x = range(olen)
#y1 = x_raw_full[d_sel,:olen,f_sel]
##y2 = x_raw_full_o[d_sel,10592:10592+olen,f_sel]
#plt.plot(x,y1)
##plt.plot(x,y2)
#
#
#x = range(olen)
#y1 = obs_tr[d_sel,:olen]
##y1 = obs_te[d_sel,:olen]
##y2 = label_o[d_sel,10592:10592+olen]
#y2 = obs_o[d_sel,10592:10592+olen]
#plt.plot(x,y1)
#plt.plot(x,y2)
#
#
## end debugging ---------------------






