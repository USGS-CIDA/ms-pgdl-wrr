# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:45:53 2018

@author: jiaxx
"""

import pandas as pd
import numpy as np

feat = pd.read_feather('fig_2/in/jordan/mendota_meteo.feather')
glm = pd.read_feather('fig_2/in/jordan/Mendota_temperatures.feather')

feat.columns
feat.values
feat.values.shape


# import previous data
x_full_o = np.load('fig_2/in/xiaowei/processed_features.npy')
x_raw_full_o = np.load('fig_2/in/xiaowei/features.npy')
diag_full_o = np.load('fig_2/in/xiaowei/diag.npy')
obs_o = np.load('fig_2/in/xiaowei/Obs_temp.npy')
label_o = np.load('fig_2/in/xiaowei/labels.npy')
dates_o = np.load('fig_2/in/xiaowei/dates.npy') # 10592

## SKIP THIS ------------ match with previous data
#new_date = feat.values[:,0]
#for i in range(dates.shape[0]):
#    if dates[i]==new_date[0]:
#        start_idx = i
#    elif dates[i]==new_date[-1]:
#        end_idx = i





# create x_full, x_raw_full, diag_full, label(glm)
x_raw_full = feat.values[1:,1:]  # start from the second day
new_dates = feat.values[1:,0]
np.save('fig_2/in/xiaowei/dates_mendota.npy',new_dates)


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

np.save('fig_2/in/xiaowei/features_mendota.npy',x_raw_full_new)
x_raw_full = np.load('fig_2/in/xiaowei/features_mendota.npy')

# standardize features
from sklearn import preprocessing
x_full = preprocessing.scale(np.reshape(x_raw_full,[n_depths*n_steps,x_raw_full.shape[-1]]))
x_full = np.reshape(x_full,[n_depths,n_steps,x_full.shape[-1]])
np.save('fig_2/in/xiaowei/processed_features_mendota.npy',x_full)


# label_glm
glm_new = glm.values[:,1:n_depths+1]
glm_new = np.transpose(glm_new)

labels = np.zeros([n_depths,n_steps],dtype=np.float64)
for i in range(n_depths):
    for j in range(n_steps):
        labels[i,j] = glm_new[i,j]

np.save('fig_2/in/xiaowei/labels_mendota.npy',labels)


# phy files ------------------------------------------------------------

#diag_all = pd.read_feather('../Generic_GLM_Mendota_diagnostics.feather')
#diag_all.columns
#idx = [-11,-10,3]
#diag_sel = diag_all.values[:,idx]
#diag_sel[:,2] = diag_sel[:,2]>0
#diag_sel = np.tile(diag_sel,[n_depths,1,1])

ice = glm.values[:,-1]
diag_sel = np.zeros([n_depths,n_steps,3])

for i in range(n_depths):
    for j in range(3):
        diag_sel[i,:,j] = ice



diag = np.zeros([n_depths, n_steps, 3], dtype=np.float64)

for i in range(n_depths):
    for j in range(n_steps):
        diag[i,j,:] = diag_sel[i,j,:]
np.save('fig_2/in/xiaowei/diag_mendota.npy',diag)



#
##x_full = np.load('processed_features.npy')
##x_raw_full = np.load('features.npy')
##diag_full = np.load('diag.npy')
##label = np.load('Obs_temp.npy')
##mask = np.load('Obs_mask.npy')
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
#y2 = label_o[d_sel,10592:10592+olen]
#plt.plot(x,y1)
#plt.plot(x,y2)
#
#
#f_sel = 8
#x = range(olen)
#y1 = x_raw_full[d_sel,:olen,f_sel]
#y2 = x_raw_full_o[d_sel,10592:10592+olen,f_sel]
#plt.plot(x,y1)
#plt.plot(x,y2)
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
#
#
#
#
#
#
#
## generate obs data and mask - copy to execution files
#
#new_dates = np.load('dates.npy')
#
#train_data = pd.read_feather('../experiment_01/mendota_training_002profiles_experiment_01.feather')
#train_data.columns
#
#tr_date = train_data.values[:,0]
#tr_depth = train_data.values[:,1]
#tr_temp = train_data.values[:,2]
#
#m_tr = np.zeros([n_depths,n_steps])
#obs_tr = np.zeros([n_depths,n_steps])
#k=0
##dd = 0
#for i in range(new_dates.shape[0]):
#    if k>=tr_date.shape[0]:
#        break
#    while new_dates[i]==tr_date[k]:
#        d = int(tr_depth[k]/0.5)
#        m_tr[d,i]=1
#        obs_tr[d,i]=tr_temp[k]
#        k+=1
#        if k>=tr_date.shape[0]:
#            break
#
#
#
#
#test_data = pd.read_feather('../experiment_01/mendota_test_experiment_01.feather')
#test_data.columns
#
#te_date = test_data.values[:,0]
#te_depth = test_data.values[:,1]
#te_temp = test_data.values[:,2]
#
#m_te = np.zeros([n_depths,n_steps])
#obs_te = np.zeros([n_depths,n_steps])
#k=0
##dd = 0
#for i in range(new_dates.shape[0]):
#    if k>=te_date.shape[0]:
#        break
#    while new_dates[i]==te_date[k]:
#        d = int(te_depth[k]/0.5)
#        if m_te[d,i]==1:
#            print(d,te_depth[k])
#        m_te[d,i]=1
#        obs_te[d,i]=te_temp[k]
#        k+=1
#        if k>=te_date.shape[0]:
#            break


