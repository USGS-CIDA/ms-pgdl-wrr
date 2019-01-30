# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:11:47 2018
only LSTM_ + PHY input    PGRNN0
@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
#import pandas as pd
import feather
#import scipy.io
#import datetime
#from datetime import date
#from sklearn.metrics import roc_auc_score

tf.reset_default_graph()
random.seed(9001)

''' Declare constants '''
learning_rate = 0.005
epochs = 400 #40 #100
#batch_size = 200
hidden_size = 20
state_size = 21 #7 
input_size = 9
phy_size = 10
npic = 9
n_steps = int(3185/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1

''' Build Graph '''
# Graph input/output
x = tf.placeholder("float", [None, n_steps, input_size]) #tf.float32
y = tf.placeholder("float", [None, n_steps]) #tf.int32
m = tf.placeholder("float", [None, n_steps])
bt_sz = tf.placeholder("int32", None) #tf.int32
x_u = tf.placeholder("float", [None, n_steps, input_size]) 
                  
# Graph weights
weights = {
    #'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([state_size, n_classes]))
}
biases = {
    #'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# embedding of each time step
#X = tf.unstack(x, n_steps, 1)
X=x
lstm_cell = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=1.0) 

state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32) #46*(500*7)
#state_series_x, current_state_x = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
# generate context vector and latent outputs for each time step
#w_c = tf.get_variable('w_c',[state_size, hidden_size], tf.float32,
#                                 tf.random_normal_initializer(stddev=0.02))
#w_h = tf.get_variable('w_h',[state_size, hidden_size], tf.float32,
#                                 tf.random_normal_initializer(stddev=0.02))
w_fin = tf.get_variable('w_fin',[state_size, n_classes], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
#b_p = tf.get_variable('b_p',[state_size],  tf.float32,
#                                 initializer=tf.constant_initializer(0.0))
b_fin = tf.get_variable('b_fin',[n_classes],  tf.float32,
                                 initializer=tf.constant_initializer(0.0))

pred=[]
#pred_s = []
for i in range(n_steps):
    tp1 = state_series_x[:,i,:]
#    tp1 = tf.reshape(state_series_x[:,i,:],[bt_sz,state_size])
    pt = tf.matmul(tp1,w_fin)+b_fin 
#    pt_s = tf.nn.softmax(pt)
    pred.append(pt)
#    pred_s.append(pt_s)

pred = tf.stack(pred,axis=1)
pred_s = tf.reshape(pred,[-1,1])
y_s = tf.reshape(y,[-1,1])
m_s = tf.reshape(m,[-1,1])

#cost = tf.sqrt(reduce_mean(tf.square(tf.substract())))
r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s-y_s),m_s)))/tf.reduce_sum(m_s))


#plos = 0
#pred_d = 1000*(1-(pred+288.9414)*((pred-3.9863)**2)/(508929.2*(pred+68.12963))) # density
#for k in range(51*npic-1):
#    if (k+1)%51!=0:
#        diff = tf.reshape((pred_d[k,:,:]-pred_d[k+1,:,:]>0),[1,n_steps])
#        diff = tf.maximum(np.zeros([1,n_steps],dtype=np.float32), tf.cast(diff,tf.float32))
#        plos = plos+tf.reduce_sum(diff)

#X_u = tf.unstack(x_u, n_steps, 1) 
#
#with tf.variable_scope("rnn", reuse=True) as scope_sp:
#    state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, X_u, dtype=tf.float32, scope=scope_sp) 
##    state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, x_u, dtype=tf.float32, scope=scope_sp) 
#
#pred_u=[]
##pred_s = []
#for i in range(n_steps):
#    tp2 = state_series_xu[i]
##    tp2 = tf.reshape(state_series_xu[:,i,:],[bt_sz,state_size])
#    pt2 = tf.matmul(tp2,w_fin)+b_fin
#    pred_u.append(pt2)
#
#pred_u = tf.stack(pred_u,axis=1)
#plos_u = 0
#pred_du = 1000*(1-(pred_u+288.9414)*((pred_u-3.9863)**2)/(508929.2*(pred_u+68.12963))) # density
#for k in range(51*npic-1):
#    if (k+1)%51!=0:
#        diff_u = tf.reshape((pred_du[k,:,:]-pred_du[k+1,:,:]>0),[1,n_steps])
#        diff_u = tf.maximum(np.zeros([1,n_steps],dtype=np.float32), tf.cast(diff_u,tf.float32))
#        plos_u = plos_u+tf.reduce_sum(diff_u)

def transformTempToDensity(temp):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    densities = 1000*(1-((temp+288.9414)*tf.pow(temp- 3.9863,2))/(508929.2*(temp+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))

    return densities

def calculate_ec_loss(inputs, outputs, phys, depth_areas, n_depths, ec_threshold, combine_days=1):
    #******************************************************
    #description: calculates energy conservation loss
    #parameters: 
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@combine_days: how many days to look back to see if energy is conserved
    #*********************************************************************************
    n_sets = N_sec#= np.floor(inputs.size()[0] / n_depths)#sets of depths in batch
    
#    diff_vec = torch.empty((inputs.size()[1]))
#    n_dates = inputs.size()[1]
    # outputs = labels
    
#    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs)
    # print("modeled densities: ", densities)

    diff_per_set = [] #torch.empty(n_sets) 
    for i in range(n_sets):
        #loop through sets of n_depths
        #indices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths

        #calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)
        #calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index,:,:], outputs[start_index,:], combine_days)
#        ### can use this to plot energy delta and flux over time to see if they line up
#        doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
#        doy = doy[1:-1]
        
#        print(lake_energy_deltas)
        diff_vec = tf.abs(lake_energy_deltas - lake_energy_fluxes) #.abs_()
        
        # sparkling og ice guesstimate
        # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

        #actual ice
#        print(diff_vec)
#        print(phys)
        tmp_mask = 1-phys[start_index+1,1:-1,9] #(phys[start_index+1,:,9] == 0)
#        print(tmp_mask)
#        print(diff_vec)
#        print(tmp_mask)
        tmp_loss = tf.reduce_mean(diff_vec*tf.cast(tmp_mask,tf.float32))
        diff_per_set.append(tmp_loss)
        
##        print(phys)
#        diff_vec = diff_vec[tf.where((phys[1:(n_depths-tf.shape(diff_vec)[0]-1),9] == 0))[0]]
#        # #compute difference to be used as penalty
#        if tf.shape(diff_vec)[0]==0: #.size() == torch.Size([0]):
#            diff_per_set.append(0) #diff_per_set[i] = 0
#        else:
#            diff_per_set.append(tf.reduce_mean(diff_vec)) #diff_per_set[i] = diff_vec.mean()

    diff_per_set_r = tf.stack(diff_per_set)
    
    diff_per_set = tf.clip_by_value(diff_per_set_r - ec_threshold, clip_value_min=0,clip_value_max=999999)
#    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    return tf.reduce_mean(diff_per_set),diff_vec,diff_per_set_r,diff_per_set #, lake_energy_deltas, lake_energy_fluxes#.mean()

def calculate_lake_energy(temps, densities, depth_areas):
    #calculate the total energy of the lake for every timestep
    #sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    #then multiply by the specific heat of water 
    dz = 0.5 #thickness for each layer, hardcoded for now
    cw = 4186 #specific heat of water
#    energy = torch.empty_like(temps[0,:])
    n_depths = 37
#    depth_areas = depth_areas.view(n_depths,1).expand(n_depths, temps.size()[1])

#    print(depth_areas)
    depth_areas = tf.reshape(depth_areas,[n_depths,1])
    energy = tf.reduce_sum(tf.multiply(tf.cast(depth_areas,tf.float32),temps)*densities*dz*cw,0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    #given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
#    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400 #seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1])/time/surface_area
#    energy_deltas = (energies[1:] - energies[:-1])/(time*surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas


def calculate_air_density(air_temp, rh):
    #returns air density in kg / m^3
    #equation from page 13 GLM/GLEON paper(et al Hipsey)

    #Ratio of the molecular (or molar) weight of water to dry air
    mwrw2a = 18.016 / 28.966
    c_gas = 1.0e3 * 8.31436 / 28.966

    #atmospheric pressure
    p = 1013. #mb

    #water vapor pressure
    vapPressure = calculate_vapour_pressure_air(rh,air_temp)

    #water vapor mixing ratio (from GLM code glm_surface.c)
    r = mwrw2a * vapPressure/(p - vapPressure)
    # print( 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15)))
    # print("vs")
    # print(1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))
    # sys.exit()
    # return 0.348*(1+r)/(1+1.61*r)*(p/(air_temp+273.15))
    return (1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))*100# 
def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 22 in GLM/GLEON paper(et al Hipsey)
    #GLM code ->  Q_sensibleheat = -CH * (rho_air * 1005.) * WindSp * (Lake[surfLayer].Temp - MetData.AirTemp);
    #calculate air density 
    rho_a = calculate_air_density(air_temp, rel_hum)

    #specific heat capacity of air in J/(kg*C)
    c_a = 1005.


    #bulk aerodynamic coefficient for sensible heat transfer
    c_H = 0.0013

    #wind speed at 10m
    U_10 = calculate_wind_speed_10m(wind_speed)
    # U_10 = wind_speed
    return -rho_a*c_a*c_H*U_10*(surf_temp - air_temp)
 
def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    #equation 23 in GLM/GLEON paper(et al Hipsey)
    #GLM code-> Q_latentheat = -CE * rho_air * Latent_Heat_Evap * (0.622/p_atm) * WindSp * (SatVap_surface - MetData.SatVapDef)
    # where,         SatVap_surface = saturated_vapour(Lake[surfLayer].Temp);
    #                rho_air = atm_density(p_atm*100.0,MetData.SatVapDef,MetData.AirTemp);
    #air density in kg/m^3
    rho_a = calculate_air_density(air_temp, rel_hum)

    #bulk aerodynamic coefficient for latent heat transfer
    c_E = 0.0013

    #latent heat of vaporization (J/kg)
    lambda_v = 2.453e6

    #wind speed at 10m height
    # U_10 = wind_speed
    U_10 = calculate_wind_speed_10m(wind_speed)
# 
    #ratio of molecular weight of water to that of dry air
    omega = 0.622

    #air pressure in mb
    p = 1013.

    e_s = calculate_vapour_pressure_saturated(surf_temp)
    e_a = calculate_vapour_pressure_air(rel_hum, air_temp)
    return -rho_a*c_E*lambda_v*U_10*(omega/p)*(e_s - e_a)


def calculate_vapour_pressure_air(rel_hum, temp):
    rh_scaling_factor = 1
    return rh_scaling_factor * (rel_hum / 100) * calculate_vapour_pressure_saturated(temp)

def calculate_vapour_pressure_saturated(temp):
    # returns in miilibars
    # print(torch.pow(10, (9.28603523 - (2332.37885/(temp+273.15)))))

    #Converted pow function to exp function workaround pytorch not having autograd implemented for pow
    exponent = (9.28603523 - (2332.37885/(temp+273.15))*np.log(10))
    return tf.exp(exponent)

def calculate_wind_speed_10m(ws, ref_height=2.):
    #from GLM code glm_surface.c
    c_z0 = 0.001 #default roughness
    return ws*(tf.log(10.0/c_z0)/tf.log(ref_height/c_z0))


def calculate_energy_fluxes(phys, surf_temps, combine_days):
    # print("surface_depth = ", phys[0:5,1])
#    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    # E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    # H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
#    time = 86400 #seconds per day
#    surface_area = 39865825 
    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_arr = phys[:-1,3] + (phys[1:,3]-phys[:-1,3])/2
    R_lw_out_arr = e_s*sigma*(tf.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,4] 
    air_temp2 = phys[1:,4]
    rel_hum = phys[:-1,5]
    rel_hum2 = phys[1:,5]
    ws = phys[:-1, 6]
    ws2 = phys[1:,6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

#    #test
#    print(R_sw_arr)
#    print(R_lw_arr)
#    print(R_lw_out_arr)
#    print(E)
#    print(H)
#    fluxes = R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1]
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])
    return fluxes


unsup_inputs = tf.placeholder("float", [None, n_steps, input_size]) #tf.float32

with tf.variable_scope("rnn", reuse=True) as scope_sp:
    state_series_xu, current_state_xu = tf.nn.dynamic_rnn(lstm_cell, unsup_inputs, dtype=tf.float32, scope=scope_sp) 

pred_u=[]
#pred_s = []
for i in range(n_steps):
    tp2 = state_series_xu[:,i,:]
    pt2 = tf.matmul(tp2,w_fin)+b_fin
    pred_u.append(pt2)

pred_u = tf.stack(pred_u,axis=1)
pred_u = tf.reshape(pred_u,[-1,n_steps])


unsup_phys_data = tf.placeholder("float", [None, n_steps, phy_size]) #tf.float32
depth_areas = np.array([637641.569,637641.569,592095.7426,592095.7426,546549.9163,546549.9163,546549.9163,501004.0899,501004.0899,
                        501004.0899, 455458.2636,455458.2636,409912.4372,409912.4372,409912.4372,364366.6109,364366.6109,
                        318820.7845,318820.7845,318820.7845,273274.9581,273274.9581,273274.9581,227729.1318,227729.1318,
                        182183.3054,182183.3054,182183.3054,136637.4791,136637.4791,136637.4791,91091.65271,91091.65271,
                        45545.82636,45545.82636,45545.82636,0])
n_depths = 37
ec_threshold = 24


unsup_loss,a,b,c = calculate_ec_loss(unsup_inputs,
                                       pred_u,
                                       unsup_phys_data,                                     
                                       depth_areas,
                                       n_depths,
                                       ec_threshold,
                                       combine_days=1)


plam = 0.15
elam = 0.005
cost = r_cost + elam*unsup_loss#+plam*plos+plam*plos_u

#cost = tf.reduce_mean(tf.nn.(labels = y, logits = pred)) # + l2 # Softmax loss
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
tvars = tf.trainable_variables()
for i in tvars:
    print(i)
saver = tf.train.Saver(max_to_keep=3)
grads = tf.gradients(cost, tvars)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars))


# load data ---------------------------------------------------------------
x_full = np.load('processed_features_sparkling.npy')
x_raw_full = np.load('features_sparkling.npy')
diag_full = np.load('diag_sparkling.npy')

phy_full = np.concatenate((x_raw_full[:,:,:-2],diag_full),axis=2)


# training and testing ------------------------------------------------------
new_dates = np.load('dates_sparkling.npy')

train_data = feather.read_dataframe('../data/Trainingdata/sparkling_season_training_500_profiles_experiment_01.feather')
#train_data.columns

tr_date = train_data.values[:,0]
tr_depth = train_data.values[:,1]
tr_temp = train_data.values[:,2]

t_steps = 3185
m_tr = np.zeros([n_depths,t_steps])
obs_tr = np.zeros([n_depths,t_steps])
k=0
#dd = 0
for i in range(new_dates.shape[0]):
    for k in range(tr_date.shape[0]):
        if new_dates[i]==tr_date[k]:
            d = min(int(tr_depth[k]/0.5),n_depths-1)
            m_tr[d,i]=1
            obs_tr[d,i]=tr_temp[k]
            k+=1
            if k>=tr_date.shape[0]:
                break

m_tr[np.isnan(obs_tr)]=0
obs_tr[np.isnan(obs_tr)]=0   

test_data = feather.read_dataframe('../data/Testdata/sparkling_season_test.feather')
#test_data.columns

te_date = test_data.values[:,0]
te_depth = test_data.values[:,1]
te_temp = test_data.values[:,2]

m_te = np.zeros([n_depths,t_steps])
obs_te = np.zeros([n_depths,t_steps])
k=0
#dd = 0
for i in range(new_dates.shape[0]):
    for k in range(te_date.shape[0]):
        if new_dates[i]==te_date[k]:
            d = min(int(te_depth[k]/0.5),n_depths-1)
    #        if m_te[d,i]==1:
    #            print(d,te_depth[k])
            m_te[d,i]=1
            obs_te[d,i]=te_temp[k]
            k+=1
            if k>=te_date.shape[0]:
                break
            
m_te[np.isnan(obs_te)]=0
obs_te[np.isnan(obs_te)]=0

x_train = np.zeros([37*N_sec,n_steps,input_size])
y_train = np.zeros([37*N_sec,n_steps])
p_train = np.zeros([37*N_sec,n_steps,phy_size])
m_train = np.zeros([37*N_sec,n_steps])
y_test = np.zeros([37*N_sec,n_steps])
m_test = np.zeros([37*N_sec,n_steps])



for i in range(1,N_sec+1):
    x_train[(i-1)*37:i*37,:,:]=x_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    y_train[(i-1)*37:i*37,:]=obs_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    p_train[(i-1)*37:i*37,:,:]=phy_full[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2),:]
    m_train[(i-1)*37:i*37,:]=m_tr[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    
    y_test[(i-1)*37:i*37,:]=obs_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]
    m_test[(i-1)*37:i*37,:]=m_te[:,int((i-1)*n_steps/2):int((i+1)*n_steps/2)]


x_f = x_train
p_f = p_train
# finish loading data --------------------------------------------------------

#total_batch = int(x_train.shape[0]/batch_size)
merr = 20
metr = 20
ploss=10000
using_pretrained=1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # if using pretrained model
    if using_pretrained==1:
        saver.restore(sess, "./pretrained_model/model_sparkling449_b.ckpt")

        
    for epoch in range(epochs):
#        
#        for i in range(total_batch - 1): # better code?
        _, loss,rc,ec,aa,bb,cc,prd1 = sess.run(
                [train_op, cost,r_cost,unsup_loss,a,b,c,pred],
                feed_dict = {
                        x: x_train,
                        y: y_train,
                        m: m_train,
                        unsup_inputs: x_f,
                        unsup_phys_data: p_f,
                        bt_sz: 37*N_sec
            })
        
        if epoch%1==0:
            print("Step " + str(epoch) + ", BatLoss= " + \
              "{:.4f}".format(loss) + ", Rc= " + \
              "{:.4f}".format(rc) + ", Ec= " + \
              "{:.4f}".format(ec))
        
        
        if epoch>30:
            loss_te,prd = sess.run([r_cost,pred], feed_dict = {x: x_train, y: y_test, m: m_test, bt_sz: 37*N_sec})
            
            if merr>loss_te:
                merr=loss_te
                metr=rc
#                save_path = saver.save(sess, "./model"+str(epoch)+"_EC_sparkling.ckpt") #sparkling2 is just a different trial
#                print("Model saved in path: %s" % save_path)
##                np.save('pred_EC_sparkling.npy',prd)
#                np.save('pred_EC_sparkling_tr1.npy',prd1)
#                np.save('pred_EC_sparkling_tr2.npy',prd2)
#               
                # save the predicted outputs
                prd_o = np.zeros([n_depths,3185])
                prd_o[:,:353] = prd[0:n_depths,:,0]
                
                for j in range(N_sec-1):   # 18*125    +250 = 2500
                    st_idx = 353-(int((j+1)*353/2)-int(j*353/2))
                    prd_o[:, 353+int(j*353/2):353+int((j+1)*353/2)] = prd[(j+1)*n_depths:(j+2)*n_depths,st_idx:,0]
                
                # calculate RMSE
                ttt = np.sum(m_te)
                rm = np.sqrt(np.sum(np.power((prd_o-obs_te)*m_te,2))/ttt)
                np.save('PGRNN_season_sparkling_exp1.npy',prd_o)
            print("RMSE of processed file: " + "{:.4f}".format(rm))
 ## calculate phy-loss
#                ploss = 0
#                prd_f = prd
#                prd_d = 1000*(1-(prd+288.9414)*((prd-3.9863)**2)/(378929.2*(prd+68.12963))) # density
#                for k in range(51*npic-1):
#                    if (k+1)%51!=0:
#                        dif = np.reshape((prd_d[k,:,:]-prd_d[k+1,:,:]>0),[1,n_steps])
#                        dif = np.maximum(np.zeros([1,n_steps]),dif)
#                        ploss = ploss+np.sum(dif)
#                ploss = ploss/n_steps/npic/37
                
            print("Loss_te " + \
                  "{:.4f}".format(loss_te) + " Min_loss_te " + \
                  "{:.4f}".format(merr) + " Min_loss_tr " + \
                  "{:.4f}".format(metr)) #+ " Phy_loss " + \
#                  "{:.4f}".format(ploss) ) 
#            if ploss<0.0001:
#                    break

