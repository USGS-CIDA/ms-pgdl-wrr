# -*- coding: utf-8 -*-
"""
Implementation of the energy balance component of the custom loss function
"""

import numpy as np
import tensorflow as tf

def calculate_ec_loss(inputs, outputs, phys, depth_areas, n_depths, ec_threshold, n_sets):
    """Calculate energy conservation loss

    Args:
        inputs (n_depths*n_sets, n_days, 8): Features (standardized) of sw_radiation, lw_radiation, etc
        outputs (n_depths*n_sets, n_days): Labels predicted by the neural network during this training iteration. nrow is n_depths*n_sets; ncol is sequence length
        phys (n_depths*n_sets, n_days, 8): Features (not standardized) of sw_radiation, lw_radiation, etc
        depth_areas (n_depths): Cross-sectional areas ordered by depth layer, shallow to deep
        n_depths (1): Number of depths (= length of depth_areas) (= number of output rows / n_sets)
        ec_threshold (1): Energy imbalance threshold below which we won't penalize
        n_sets (1): Number of sets of depths in the batch

    Returns:
        (1) Energy imbalance exceeding the ec_threshold as an average over all timesteps in the outputs (W/m2)
    """

    densities = transform_temp_to_density(outputs)

    # Loop through sets of n_depths (time series sequences of fixed length)
    diff_per_set = []
    for i in range(n_sets):
        # Identify row indices for set i within the outputs, phys, etc. matrices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths

        # Calculate lake energy for each timestep (vector, J)
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)

        # Calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:] # the first delta is nonsense

        # Calculate sum of energy flux into or out of the lake at each timestep
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index, :, :], outputs[start_index, :])

        # Calculate absolute values of energy imbalances at each timestep
        diff_vec = tf.abs(lake_energy_deltas - lake_energy_fluxes)

        # Mask dates with ice, for which we won't require energy conservation
        tmp_mask = 1-phys[start_index+1, 1:-1, 9] # the boolean mask for ice is the 9th column in the phys matrix
        tmp_loss = tf.reduce_mean(diff_vec * tf.cast(tmp_mask, tf.float32)) # apply the mask to diff_vec, then take mean

        # Add the energy imbalance for this set to the vector of imbalances
        diff_per_set.append(tmp_loss)

    # combine the diff_per_set vectors in to a matrix (each vector is a row)
    diff_per_set_r = tf.stack(diff_per_set)

    # reduce the penalty for every set by ec_threshold and then only penalize
    # at all those sets with a mean diff_per_set greater than ec_threshold
    diff_per_set = tf.clip_by_value(diff_per_set_r - ec_threshold, clip_value_min=0, clip_value_max=999999)

    # return the mean of the penalties (or non-penalties) over all the sets
    return tf.reduce_mean(diff_per_set)

def transform_temp_to_density(temp):
    """Convert water temperature to density

    Args:
        temp (n_depths*n_sets, n_days): Water temperatures (degC)

    Returns:
        (n_depths*n_sets, n_days) Water densities (kg/m3)
    """
    densities = 1000 * (1 - ((temp + 288.9414) * tf.pow(temp - 3.9863, 2)) / (508929.2 * (temp + 68.12963)))
    return densities

def calculate_lake_energy(temps, densities, depth_areas):
    """Calculate the total energy of the lake for every timestep

    Args:
        temps (n_depths, n_days): Water temperatures (degC)
        densities (n_depths, n_days): Water densities (kg/m3)
        depth_areas (n_depths): Lake cross-sectional area at each depth (m2)

    Returns:
        (n_days) Total lake energies (J)
    """
    # Format depth_areas into a column tensor (m2)
    depth_areas_col = tf.cast(tf.reshape(depth_areas, [depth_areas.size, 1]), tf.float32)

    # Compute energy in each layer for each timestep (result is in J)
    dz = 0.5 # thickness for each depth layer (m)
    cw = 4186 # specific heat of water (J/(kg degC))
    energy_mat = tf.multiply(depth_areas_col, temps)*densities*dz*cw

    # Total lake energy by timestep is sum of over all layers in the depth
    # profile at that timestep
    energy = tf.reduce_sum(energy_mat, 0)
    return energy

def calculate_lake_energy_deltas(energies, surface_area):
    """Compute the differences in lake energies between timestep pairs as W/m2

    Args:
        energies (n_days): Lake energies at a series of timesteps (J)
        surface_area (1): Area of the surface of the lake (m2)

    Returns:
        (n_days - 1) Energy change by timestep (W/m2)
    """
    # Calculate daily differences and convert from J = W*s to W/m2
    seconds_per_timestep = 86400
    energy_deltas = (energies[1:] - energies[:-1]) / seconds_per_timestep / surface_area
    return energy_deltas

def calculate_air_density(air_temp, rel_hum):
    """Calculate density of air

    Note:
        Equation from page 480 of Hipsey et al. (2019):
        rho_a = 0.348*(1+r)/(1+1.61*r)*p/Ta

    Args:
        air_temp (n_days): Air temperatures in degC
        rel_hum (n_days): Relative humidities (%)

    Returns:
        (n_days) Air density (kg/m3)
    """
    # Prepare components for the air density equation
    mwrw2a = 18.016 / 28.966 # ratio of the molecular (or molar) weight of water to dry air
    c_gas = 1.0e3 * 8.31436 / 28.966 # 100./c_gas = 0.348
    p = 1013.0 # atmospheric pressure (mb)
    vapPressure = calculate_vapour_pressure_air(rel_hum, air_temp) # water vapor pressure
    r = mwrw2a * vapPressure/(p - vapPressure) # water vapor mixing ratio

    # The following is equivalent to 0.348*(1+r)/(1+1.61*r)*p/Ta
    # where 100./c_gas = 0.348, 1./mwrw2a=1.61, and
    # Ta is air_temp in Kelvin = (air_temp + 273.15)
    return 100.*(1.0/c_gas * (1 + r)/(1 + r/mwrw2a) * p/(air_temp + 273.15))

def calculate_heat_flux_sensible(surf_temp, air_temp, rel_hum, wind_speed):
    """Calculate sensible heat flux

    Note:
        Equation 22 in Hipsey et al. (2019):
        phi_H = - rho_a * c_a * C_H * U_10 * (T_s - T_a)

    Args:
        surf_temp (n_days): Water surface temperatures (degC)
        air_temp (n_days): Air temperatures (degC)
        rel_hum (n_days): Relative humidities (%)
        wind_speed (n_days): Wind speeds at 2m above lake surface (m/s)

    Returns:
        (n_days) Sensible heat flux (W/m2)
    """
    # Prepare components for the sensible heat flux equation
    rho_a = calculate_air_density(air_temp, rel_hum) # air density (kg/m3)
    c_a = 1005. # specific heat capacity of air in J/(kg*degC)
    C_H = 0.0013 # bulk aerodynamic coefficient for sensible heat transfer (unitless)
    U_10 = calculate_wind_speed_10m(wind_speed) # wind speed at 10m height (m/s)

    # (kg/m3) * J/(kg*degC) * 1 * (m/s) * degC = (J/s)/m2 = W/m2
    return -rho_a * c_a * C_H * U_10 * (surf_temp - air_temp)

def calculate_heat_flux_latent(surf_temp, air_temp, rel_hum, wind_speed):
    """Calculate latent heat flux

    Note:
        Equation 23 in Hipsey et al. (2019):
        phi_E = - rho_a * C_E * lambda_v * U_10 * (omega / p) * (e_s[T_s] - e_a[T_a])

    Args:
        surf_temp (n_days): Water surface temperatures (degC)
        air_temp (n_days): Air temperatures (degC)
        rel_hum (n_days): Relative humidities (%)
        wind_speed (n_days): Wind speeds at 2m above lake surface (m/s)

    Returns:
        (n_days) Latent heat flux (W/m2)
    """
    # Prepare components for the latent heat flux equation
    rho_a = calculate_air_density(air_temp, rel_hum) # air density (kg/m3)
    C_E = 0.0013 # bulk aerodynamic coefficient for latent heat transfer (unitless)
    lambda_v = 2.453e6 # latent heat of vaporization (J/kg)
    U_10 = calculate_wind_speed_10m(wind_speed) # wind speed at 10m height (m/s)
    omega = 0.622 # ratio of molecular weight of water to that of dry air (kg/kg)
    p = 1013. # air pressure (mb)
    e_s_T_s = calculate_vapour_pressure_saturated(surf_temp) # saturated vapor pressure (mb)
    e_a_T_a = calculate_vapour_pressure_air(rel_hum, air_temp) # vapor pressure (mb)

    # (kg/m3) * 1 * (J/kg) * (m/s) * (kg/kg)/mb * (mb-mb) = (J/s)/m2 - W/m2
    return -rho_a * C_E * lambda_v * U_10 * (omega/p) * (e_s_T_s - e_a_T_a)

def calculate_vapour_pressure_air(rel_hum, air_temp):
    """Calculate the vapour pressure of air

    Note:
        Equation 24 of Hipsey et al. (2019):
        e_a[T_a] = (f_RH * RH_x / 100) * e_s[T_s]

    Args:
        rel_hum (n_days): Relative humidities (%)
        air_temp (n_days): Air temperatures (degC)

    Returns:
        (n_days) Vapour pressures of air (mb)
    """
    f_RH = 1 # relative humidity scaling factor (unitless)
    e_s_T_a = calculate_vapour_pressure_saturated(air_temp) # saturated vapor pressure (mb)

    # units check: 1 * % / 1 * mb = mb
    return f_RH * (rel_hum / 100) * e_s_T_a

def calculate_vapour_pressure_saturated(temp):
    """Calculate the saturated vapour pressure in the air
    near or above the water surface

    Note:
        Equation 24 of Hipsey et al. (2019):
        e_s[T_s] = 10^(9.28603523 - (2332.37885/(temp + 273.15)))

    Args:
        temp (n_days): Air temperatures, assumed to equal water temperatures
            at air-water interface (degC)

    Returns:
        (n_days) Saturated vapour pressures of air (mb)
    """
    #return tf.exp((9.28603523 - (2332.37885/(temp + 273.15)) * np.log(10))) # wrong
    return tf.exp((9.28603523 - (2332.37885/(temp + 273.15))) * np.log(10)) # right

def calculate_wind_speed_10m(ws, ref_height=2.):
    """Estimate the wind speed at 10m above the lake surface

    Note:
        Equation in glm_surface.c of GLM source code:
        U10 = WindSp * (log(10.0/c_z0)/log(WIND_HEIGHT/c_z0));

    Args:
        ws (n_days): Wind speeds at reference height (m/s)
        ref_height (1): Reference height (m)

    Returns:
        (n_days) Wind speeds at 10m above lake surface (m/s)
    """
    c_z0 = 0.001 # default roughness from glm_surface.c
    return ws * (tf.log(10.0/c_z0) / tf.log(ref_height/c_z0))

def calculate_energy_fluxes(phys, surf_temps):
    """Calculate net energy influx to lake between each pair of adjacent time points

    Args:
        phys (n_depths, n_days, n_vars): Physical data, where the variables
            are DayOfYear, Zeros, ShortWave, LongWave, AirTemp, RelHum,
            WindSpeed, Rain, Snow
        surf_temps (n_depths, n_days): Temperatures at water surface

    Returns:
        (n_days) Net energy influx to lake at each timestep (W/m2)
    """
    # Define constants
    e_s = 0.985 # emissivity of water
    alpha_sw = 0.07 # shortwave albedo
    alpha_lw = 0.03 # longwave albedo
    sigma = 5.67e-8 # Stefan-Boltzmann constant

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

    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])
    return fluxes
