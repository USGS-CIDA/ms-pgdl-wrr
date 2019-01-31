# -*- coding: utf-8 -*-
"""
Using Mendota data
"""

import numpy as np
import tensorflow as tf
import sys
import feather

## read and prepare some test data (using fig 2 mendota)

n_steps = 353 # steps per timeseries sequence

x_full = np.load('fig_2/in/xiaowei/processed_features_mendota.npy')
x_raw_full = np.load('fig_2/in/xiaowei/features_mendota.npy')
diag_full = np.load('fig_2/in/xiaowei/diag_mendota.npy')
phy_full = np.concatenate((x_raw_full[:,:,:-2],diag_full),axis=2)

# standardized input in 2 sets of sequences
Pz = np.vstack((
        x_full[:, 0:n_steps, :],
        x_full[:, n_steps:n_steps*2, :]))

# raw (non-standardized) input in 2 sets of sequences
P = np.vstack((
        phy_full[:, 0:n_steps, :],
        phy_full[:, n_steps:n_steps*2, :]))

# example predictions from PGDL
pgdl_preds = np.load('fig_2/out/PGRNN_season_mendota_exp1.npy')
y = np.vstack((
        pgdl_preds [:, 0:n_steps],
        pgdl_preds [:, n_steps:n_steps*2]))

# function to visualize a matrix with colorbar
def plot_mat(mat):
    import matplotlib.pyplot as plt
    %matplotlib qt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.subplot(111)
    im = plt.imshow(mat)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

# depth_areas are interpolated from pd.read_csv('fig_2/in/jordan/mendota_geometry.csv')
depth_areas = np.array([
    39865825,38308175,38308175,35178625,35178625,33403850,31530150,31530150,30154150,30154150,29022000,
    29022000,28063625,28063625,27501875,26744500,26744500,26084050,26084050,25310550,24685650,24685650,
    23789125,23789125,22829450,22829450,21563875,21563875,20081675,18989925,18989925,17240525,17240525,
    15659325,14100275,14100275,12271400,12271400,9962525,9962525,7777250,7777250,5956775,4039800,4039800,
    2560125,2560125,820925,820925,216125])
n_depths = depth_areas.size
ec_threshold = 24.


## build a minimal tensorflow graph for testing custom_loss functions

tf.reset_default_graph()

# graph parameters
input_size = Pz.shape[2]
n_sets = 2

# input placeholders
unsup_inputs = tf.placeholder(tf.float32, [None, n_steps, Pz.shape[2]]) # standardized features
unsup_phys_data = tf.placeholder(tf.float32, [None, n_steps, P.shape[2]]) # raw (not-standardized) features
unsup_preds = tf.placeholder(tf.float32, [None, n_steps]) # temperature predictions (usually from the NN, but here we'll use train_data)

# load the custom_loss module
sys.path.append('text_S4')
import custom_loss

# parse the physics data into labeled variables
air_temp = unsup_phys_data[1, :, 4]
rel_hum = unsup_phys_data[1, :, 5]
wind_speed_Xm = unsup_phys_data[1, :, 6]
surf_temp = unsup_preds[0, :]

# unit tests, ~one per function in custom_loss.py
densities = custom_loss.transform_temp_to_density(unsup_preds)
with tf.Session() as sess:
    densities_out = sess.run(densities, feed_dict = {unsup_preds: y})
    plot_mat(densities_out)

lake_energy = custom_loss.calculate_lake_energy(
        unsup_preds,
        densities,
        depth_areas)
with tf.Session() as sess:
    lake_energy_out = sess.run(
        lake_energy,
        feed_dict = {
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(lake_energy_out)
    plt.xlabel('Day')
    plt.ylabel('Lake Energy (J)')

lake_energy_deltas = custom_loss.calculate_lake_energy_deltas(
        lake_energy,
        depth_areas[0])
with tf.Session() as sess:
    lake_energy_deltas_out = sess.run(
        lake_energy_deltas,
        feed_dict = {
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(lake_energy_deltas_out)
    plt.xlabel('Day')
    plt.ylabel('Energy Deltas (W/m2)')

wind_speed_10m = custom_loss.calculate_wind_speed_10m(wind_speed_Xm)
with tf.Session() as sess:
    wind_speed_Xm_out, wind_speed_10m_out = sess.run(
        [wind_speed_Xm, wind_speed_10m],
        feed_dict = {
            unsup_phys_data: P[:n_depths, :]
        })
    plt.plot(wind_speed_Xm_out, c='red')
    plt.plot(wind_speed_10m_out, c='blue')
    plt.xlabel('Day')
    plt.ylabel('Wind speed at 2m (red) & 10m (blue) (m/2)')

air_density = custom_loss.calculate_air_density(air_temp, rel_hum)
with tf.Session() as sess:
    air_density_out = sess.run(
        air_density,
        feed_dict = {
            unsup_phys_data: P[:n_depths, :]
        })
    plt.plot(air_density_out)
    plt.xlabel('Day')
    plt.ylabel('Air density (kg/m3)')

e_s_T_s = custom_loss.calculate_vapour_pressure_saturated(surf_temp)
e_s_T_a = custom_loss.calculate_vapour_pressure_saturated(air_temp)
e_a_T_a = custom_loss.calculate_vapour_pressure_air(rel_hum, air_temp)
with tf.Session() as sess:
    esTs_out, esTa_out, eaTa_out = sess.run(
        [e_s_T_s, e_s_T_a, e_a_T_a],
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(esTs_out, c='red')
    plt.plot(esTa_out, c='blue')
    plt.plot(eaTa_out, c='green')
    plt.xlabel('Day')
    plt.ylabel('Vapor pressure (mb):\nesTs (red), esTa (blue), eaTa (green)')

sens_heat_flux = custom_loss.calculate_heat_flux_sensible(
        surf_temp, air_temp, rel_hum, wind_speed_Xm)
latn_heat_flux = custom_loss.calculate_heat_flux_latent(
        surf_temp, air_temp, rel_hum, wind_speed_Xm)
with tf.Session() as sess:
    sens_heat_flux_out, latn_heat_flux_out = sess.run(
        [sens_heat_flux, latn_heat_flux],
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(sens_heat_flux_out, c='blue')
    plt.plot(latn_heat_flux_out, c='red')
    plt.xlabel('Day')
    plt.ylabel('Heat flux (W/m2): sensible (blue), latent (red)')

lake_energy_fluxes = custom_loss.calculate_energy_fluxes(
        unsup_phys_data[1, :, :],
        surf_temp)
with tf.Session() as sess:
    lake_energy_fluxes_out = sess.run(
        lake_energy_fluxes,
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(lake_energy_fluxes_out)
    plt.ylim(-500, 300)
    plt.xlabel('Day')
    plt.ylabel('Sum of energy fluxes (W/m2)')

energy_imbalances = tf.abs(lake_energy_deltas[1:] - lake_energy_fluxes)
with tf.Session() as sess:
    energy_imbalances_out = sess.run(
        energy_imbalances,
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(energy_imbalances_out)
    plt.xlabel('Day')
    plt.ylabel('Absolute energy imbalances (deltas - fluxes, W/m2)')

ice_mask = tf.cast(1-unsup_phys_data[1, 1:-1, 9], tf.float32)
masked_energy_imbalances = energy_imbalances * ice_mask
with tf.Session() as sess:
    masked_energy_imbalances_out = sess.run(
        masked_energy_imbalances,
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    plt.plot(masked_energy_imbalances_out)
    plt.ylim(-10,475)
    plt.xlabel('Day')
    plt.ylabel('Masked abs energy imbalances (deltas - fluxes, W/m2)')

ec_loss_1set = tf.reduce_mean(masked_energy_imbalances)
with tf.Session() as sess:
    ec_loss_1set_out = sess.run(
        ec_loss_1set,
        feed_dict = {
            unsup_phys_data: P[:n_depths, :],
            unsup_preds: y[:n_depths, :]
        })
    print(ec_loss_1set_out)

ec_loss = custom_loss.calculate_ec_loss(
    unsup_inputs,
    unsup_preds,
    unsup_phys_data,
    depth_areas,
    n_depths,
    ec_threshold,
    n_sets)
with tf.Session() as sess:
    ec_loss_out = sess.run(
        [ec_loss],
        feed_dict = {
            unsup_inputs: Pz,
            unsup_preds: y,
            unsup_phys_data: P
        })
    print(ec_loss_out)
# with wrong calculate_vapour_pressure_saturated: 54.15
# with right calculate_vapour_pressure_saturated: 40.33
