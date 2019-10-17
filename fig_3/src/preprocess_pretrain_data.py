import pandas as pd
import numpy as np
import os
import sys
import re
import math
import shutil
from datetime import datetime
import argparse

from scipy import interpolate

parser = argparse.ArgumentParser()
parser.add_argument('--lake_name')
parser.add_argument('--met_file')
parser.add_argument('--glm_file')
parser.add_argument('--ice_file')
parser.add_argument('--raw_data_path')
parser.add_argument('--processed_path')
args = parser.parse_args()


###################################################################################
# (Jared) Jan 2019 - Read/format data for many lakes sent by Jordan (69 lakes)
###################################################################################
first_time = False
lnames = set()
n_features = 7
n_lakes = 0
if first_time:
    os.mkdir("../../data/processed/WRR_69Lake")
    os.mkdir("../../models/WRR_69Lake")

for filename in os.listdir(args.raw_data_path):
    # parse lakename from file
    m = re.search(r'^nhd_(\d+)_test_train.*', filename)
    if m is None:
        continue
    name = m.group(1)
    if name not in lnames:
        # for each unique lake
        name = name
        lnames.add(name)
        n_lakes += 1

        ############################################
        # read/format meteorological data for numpy
        #############################################
        meteo_dates = np.loadtxt('../../data/raw/figure3/nhd_' + name + '_meteo.csv', delimiter=',', dtype=np.string_,
                                 usecols=0)

        # lower/uppur cutoff indices (to match observations)
        obs = pd.read_feather('../../data/raw/figure3/nhd_' + name + '_test_train.feather')
        start_date = "{:%Y-%m-%d}".format(obs.values[0, 1]).encode()
        end_date = "{:%Y-%m-%d}".format(obs.values[-1, 1]).encode()
        lower_cutoff = np.where(meteo_dates == start_date)[0][0]  # 457
        if len(np.where(meteo_dates == end_date)[0]) < 1:
            print("observation beyond meteorological data! data will only be used up to the end of meteorological data")
            upper_cutoff = meteo_dates.shape[0]
        else:
            upper_cutoff = np.where(meteo_dates == end_date)[0][0] + 1  # 14233

        meteo_dates = meteo_dates[lower_cutoff:upper_cutoff]

        # read from file and filter dates
        meteo = np.genfromtxt('../../data/raw/figure3/nhd_' + name + '_meteo.csv', delimiter=',',
                              usecols=(1, 2, 3, 4, 5, 6, 7))
        meteo = meteo[lower_cutoff:upper_cutoff, :]

        # normalize data
        meteo_means = [meteo[:, a].mean() for a in range(n_features)]
        meteo_std = [meteo[:, a].std() for a in range(n_features)]
        meteo_norm = (meteo - meteo_means[:]) / meteo_std[:]

        # meteo = final features sans depth
        # meteo_norm = normalized final features sans depth

        ################################################################################
        # read/format GLM temperatures and observation data for numpy
        ###################################################################################

        glm_temps = pd.read_feather('../../data/raw/figure3/nhd_' + name + '_temperatures.feather')
        glm_temps = glm_temps.values[:]
        n_total_dates = glm_temps.shape[0]

        # define depths from glm file
        n_depths = glm_temps.shape[1] - 2  # minus date and ice flag
        # print("n_depths: " + str(n_depths))
        max_depth = 0.5 * (n_depths - 1)
        depths = np.arange(0, max_depth + 0.5, 0.5)
        depths_normalized = np.divide(depths - depths.mean(), depths.std())

        # format date to string to match
        glm_temps[:, 0] = np.array([glm_temps[a, 0].strftime('%Y-%m-%d') for a in range(n_total_dates)])

        if len(np.where(glm_temps[:, 0] == start_date.decode())[0]) < 1:
            print("observations begin at " + start_date.decode() + "which is before GLM data which begins at " +
                  glm_temps[0, 0])
            lower_cutoff = 0
            new_meteo_lower_cutoff = np.where(meteo_dates == glm_temps[0, 0].encode())[0][0]
            meteo = meteo[new_meteo_lower_cutoff:, :]
            meteo_norm = meteo_norm[new_meteo_lower_cutoff:, :]
            meteo_dates = meteo_dates[new_meteo_lower_cutoff:]
        else:
            lower_cutoff = np.where(glm_temps[:, 0] == start_date.decode())[0][0]

        if len(np.where(glm_temps[:, 0] == end_date.decode())[0]) < 1:
            print("observations extend to " + end_date.decode() + "which is beyond GLM data which extends to " +
                  glm_temps[-1, 0])
            upper_cutoff = glm_temps[:, 0].shape[0]
            new_meteo_upper_cutoff = np.where(meteo_dates == glm_temps[-1, 0].encode())[0][0]
            meteo = meteo[:new_meteo_upper_cutoff + 1, :]
            meteo_norm = meteo_norm[:new_meteo_upper_cutoff + 1, :]
            meteo_dates = meteo_dates[:new_meteo_upper_cutoff + 1]


        else:
            upper_cutoff = np.where(glm_temps[:, 0] == end_date.decode())[0][0]

            # format glm dates to compare for when to cutoff test set drivers in glm
        last_tst_date = obs.values[math.floor(obs.shape[0] / 3), 1]  # get last test date
        glm_dates = [np.datetime64(glm_temps[i, 0]) for i in
                     range(glm_temps.shape[0])]  # get date values in comparable form
        last_tst_date_stamp = np.datetime64(last_tst_date)  # get last test date in comparable form
        lower_cutoff_pretrain = np.where(glm_dates[:] == last_tst_date_stamp)[0][0]  # find cutoff point in glm data
        meteo_dates_np = [np.datetime64(d) for d in meteo_dates]  # convert meteo dates to comparable form
        meteo_lower_cutoff_pretrain = np.where(meteo_dates_np == last_tst_date_stamp)[0][
            0]  # cutoff point where meteo data can begin for train data
        meteo_pt = meteo[meteo_lower_cutoff_pretrain:]  # subset of meteo data AFTER the test set
        meteo_norm_pt = meteo_norm[meteo_lower_cutoff_pretrain:]  # subset of meteo data AFTER the test set
        meteo_dates_pt = meteo_dates[meteo_lower_cutoff_pretrain:]  # subset of meteo data AFTER the test set

        print("cutoff ind for glm  (to avoid test set drivers): ", lower_cutoff_pretrain)
        glm_temps_pretrain = glm_temps[lower_cutoff_pretrain:upper_cutoff + 1, :]
        glm_temps = glm_temps[lower_cutoff:upper_cutoff + 1, :]
        n_dates_pretrain = glm_temps_pretrain.shape[0]
        n_dates_meteo = meteo.shape[0]
        assert meteo_pt.shape[0] == meteo_norm_pt.shape[0]

        if n_dates_meteo != meteo.shape[0]:
            print(n_dates)
            print(meteo.shape[0])

        assert n_dates_meteo == meteo_norm.shape[0]
        assert n_dates_pretrain == meteo_pt.shape[0]
        assert n_dates_pretrain == meteo_norm_pt.shape[0]
        assert n_dates_pretrain == meteo_dates_pt.shape[0]
        # assert dates line up
        # assert(glm_temps[0,0] == meteo_dates[0].decode())

        if glm_temps[-1, 0] != meteo_dates[-1].decode():
            print(glm_temps[-1, 0])
            print(meteo_dates[-1].decode())

        assert (glm_temps[-1, 0] == meteo_dates[-1].decode())

        ice_flag = glm_temps[:, -1]
        ice_flag_pt = glm_temps_pretrain[:, -1]
        glm_temps = glm_temps[:, 1:-1]
        glm_temps_pretrain = glm_temps_pretrain[:, 1:-1]
        obs = obs.values[:, 1:]  # remove needless nhd column
        n_obs = obs.shape[0]

        ############################################################
        # fill numpy matrices
        ##################################################################
        feat_mat = np.empty((n_depths, n_dates_meteo, n_features + 2))  # [depth->7 meteo features-> ice flag]
        feat_mat[:] = np.nan
        feat_pt_mat = np.empty((n_depths, n_dates_pretrain, n_features + 2))  # [depth->7 meteo features-> ice flag]
        feat_pt_mat[:] = np.nan
        feat_norm_mat = np.empty(
            (n_depths, n_dates_meteo, n_features + 1))  # [standardized depth -> 7 std meteo features]
        feat_norm_mat[:] = np.nan
        feat_norm_pt_mat = np.empty(
            (n_depths, n_dates_pretrain, n_features + 1))  # [standardized depth -> 7 std meteo features]
        feat_norm_pt_mat[:] = np.nan
        glm_mat = np.empty((n_depths, n_dates_meteo))
        glm_mat[:] = np.nan
        glm_mat_pt = np.empty((n_depths, n_dates_pretrain))
        glm_mat_pt[:] = np.nan
        obs_trn_mat = np.empty((n_depths, n_dates_meteo))
        obs_trn_mat[:] = np.nan
        obs_tst_mat = np.empty((n_depths, n_dates_meteo))
        obs_tst_mat[:] = np.nan

        for d in range(n_depths):
            feat_mat[d, :, 0] = depths[d]
            feat_pt_mat[d, :, 0] = depths[d]
            feat_norm_mat[d, :, 0] = depths_normalized[d]
            feat_norm_pt_mat[d, :, 0] = depths_normalized[d]
            glm_mat[d, :] = glm_temps[:, d]
            glm_mat_pt[d, :] = glm_temps_pretrain[:, d]
            feat_mat[d, :, 1:-1] = meteo[:]
            feat_mat[d, :, -1] = ice_flag[:]
            feat_pt_mat[d, :, 1:-1] = meteo_pt[:]
            feat_pt_mat[d, :, -1] = ice_flag_pt[:]
            feat_norm_mat[d, :, 1:] = meteo_norm[:]
            feat_norm_pt_mat[d, :, 1:] = meteo_norm_pt[:]

        # verify all mats filled
        if np.isnan(np.sum(feat_mat)):
            print(feat_mat)
            print("ERROR: Preprocessing failed, there is missing data feat")
            sys.exit()
        if np.isnan(np.sum(feat_pt_mat)):
            print("ERROR: Preprocessing failed, there is missing data feat pt")
            sys.exit()
        if np.isnan(np.sum(feat_norm_mat)):
            print("ERROR: Preprocessing failed, there is missing data feat norm")
            sys.exit()
        if np.isnan(np.sum(feat_norm_pt_mat)):
            print("ERROR: Preprocessing failed, there is missing data feat norm")
            sys.exit()
        if np.isnan(np.sum(glm_mat)):
            # print("Warning: there is missing data in glm output")
            for i in range(n_depths):
                for t in range(n_dates_meteo):
                    if np.isnan(glm_mat[i, t]):
                        x = depths[i]
                        xp = depths[0:(i - 1)]
                        yp = glm_mat[0:(i - 1), t]
                        f = interpolate.interp1d(xp, yp, fill_value="extrapolate")
                        glm_mat[i, t] = f(x)  # interp_temp

            assert not np.isnan(np.sum(glm_mat))
        if np.isnan(np.sum(glm_mat_pt)):
            # print("Warning: there is missing data in glm output")
            for i in range(n_depths):
                for t in range(n_dates_pretrain):
                    if np.isnan(glm_mat_pt[i, t]):
                        x = depths[i]
                        xp = depths[0:(i - 1)]
                        yp = glm_mat_pt[0:(i - 1), t]
                        f = interpolate.interp1d(xp, yp, fill_value="extrapolate")
                        glm_mat_pt[i, t] = f(x)  # interp_temp

            assert not np.isnan(np.sum(glm_mat))
        # observations, round to nearest 0.5m depth and put in train/test matrices
        obs[:, 1] = np.round((obs[:, 1] * 2).astype(np.float)) / 2  # round
        # print(depths)
        obs_g = 0
        obs_d = 0

        # get unique observation days
        unq_obs_dates = np.unique(obs[:, 0])
        print(unq_obs_dates.shape[0])
        n_unq_obs_dates = unq_obs_dates.shape
        first_tst_date = obs[0, 0]
        last_tst_date = obs[math.floor(obs.shape[0] / 3), 0]
        last_tst_obs_ind = np.where(obs[:, 0] == last_tst_date)[0][-1]

        n_tst = last_tst_obs_ind + 1
        n_trn = obs.shape[0] - n_tst

        last_train_date = obs[-1, 0]
        first_train_date = obs[last_tst_obs_ind + 1, 0]
        print("first_tst_date", first_tst_date)
        print("last_tst_date ", last_tst_date)

        # test data
        n_tst_obs_placed = 0
        n_trn_obs_placed = 0

        for o in range(0, last_tst_obs_ind + 1):
            # verify data in depth range
            if obs[o, 1] > depths[-1]:
                obs_g += 1
                # print("observation depth " + str(obs[o,1]) + " is greater than the max depth of " + str(max_depth))
                continue
            if len(np.where(meteo_dates == obs[o, 0].strftime('%Y-%m-%d').encode())[0]) < 1:
                obs_d += 1
                continue
            depth_ind = np.where(depths == obs[o, 1])[0][0]
            date_ind = np.where(meteo_dates == obs[o, 0].strftime('%Y-%m-%d').encode())[0][0]
            obs_tst_mat[depth_ind, date_ind] = obs[o, 2]
            n_tst_obs_placed += 1

        # train data
        for o in range(last_tst_obs_ind + 1, n_obs):
            if obs[o, 1] > depths[-1]:
                obs_g += 1
                # print("observation depth " + str(obs[o,1]) + " is greater than the max depth of " + str(max_depth))
                continue
            depth_ind = np.where(depths == obs[o, 1])[0][0]
            if len(np.where(meteo_dates == obs[o, 0].strftime('%Y-%m-%d').encode())[0]) < 1:
                obs_d += 1
                continue

            depth_ind = np.where(depths == obs[o, 1])[0][0]
            date_ind = np.where(meteo_dates == obs[o, 0].strftime('%Y-%m-%d').encode())[0][0]

            obs_trn_mat[depth_ind, date_ind] = obs[o, 2]
            n_trn_obs_placed += 1

        d_str = ""
        if obs_d > 0:
            d_str = ", and " + str(
                obs_d) + " observations outside of combined date range of meteorological and GLM output"
        # if obs_g > 0 or obs_d > 0:
        # continue
        print("lake " + str(n_lakes) + ",  id: " + name + ": " + str(obs_g) + "/" + str(
            n_obs) + " observations greater than max depth " + str(max_depth) + d_str)
        # write features and labels to processed data
        print("training: ", first_train_date, "->", last_train_date, "(", n_trn, ")")
        print("testing: ", first_tst_date, "->", last_tst_date, "(", n_tst, ")")
        if first_time:
            os.mkdir(os.path.join(args.processed_path, name))
            os.mkdir("../../models/WRR_69Lake/" + name)
        feat_pt_path = os.path.join(args.processed_path, name, "features_pt")
        norm_feat_pt_path = os.path.join(args.processed_path, name, "processed_features_pt")
        glm_path = os.path.join(args.processed_path, name, "glm_noTest")

        # geometry
        shutil.copyfile('../../data/raw/figure3/nhd_'+name+'_geometry.csv', "../../data/processed/WRR_69Lake/"+name+"/geometry")

        np.save(feat_pt_path, feat_pt_mat)
        np.save(norm_feat_pt_path, feat_norm_pt_mat)
        np.save(glm_path, glm_mat_pt)
