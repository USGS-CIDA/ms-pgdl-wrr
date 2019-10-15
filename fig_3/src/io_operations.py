# https://github.com/jdwillard19/lake_modeling/blob/a2fb7239c929e14f6647faa21167e96ad79c0cef/src/data/io_operations.py

import pandas as pd
import os
import numpy as np
import sys
import feather

def averageTrialsToFinalOutput(lakename, realization, best_hid, best_norm='NA', trials=2, PGRNN=True):
    # for a given lake and realization of randomly chosen observations, compile the experiment results into an averaged prediction
    # @lakename = string nhd id
    # @best hid = number of best hidden units for experiment
    # @best norm (optional) = if l1 norm is used in hyperparameter optimization this is the value
    # @trials (not implemented yet) = number of trials per experiment setup
    # @PGRNN = True if PGRNN, False if RNN

    realization = str(realization)
    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'
    o_path1 = '../../scripts/manylakes/outputs' + realization + '/' + lakename + pg + '_output_' + 'nhid' + str(
        best_hid) + '_norm' + str(best_norm) + '_trial0.feather'
    o_path2 = '../../scripts/manylakes/outputs' + realization + '/' + lakename + pg + '_output_' + 'nhid' + str(
        best_hid) + '_norm' + str(best_norm) + '_trial1.feather'
    merge_path = '../../scripts/manylakes/outputs' + realization + '/' + lakename + pg + '_output_' + 'nhid' + str(
        best_hid) + '_norm' + str(best_norm) + '_BESTmerged.feather'
    merge_path2 = '../../scripts/manylakes/outputs' + realization + '/' + lakename + pg + '_output_' + 'nhid' + str(
        best_hid) + '_norm' + str(best_norm) + '_BESTmerged2.feather'
    obs1 = pd.read_feather(o_path1)
    obs2 = pd.read_feather(o_path2)
    obs_merged = obs1.copy()
    obs_merged2 = pd.DataFrame().reindex_like(obs1)  #

    obs_merged.values[:, 1:] = (obs1.values[:, 1:] + obs2.values[:, 1:]) / 2  # average the two trials
    feather.write_dataframe(obs_merged, merge_path)

    obs_merged2.values[:, 1:] = (obs1.values[:, 1:] + obs2.values[:, 1:]) / 2
    obs_merged2['date'] = pd.to_datetime(df['date'])
    # obs_merged2.values[:,0] = obs1.values[:,0]
    feather.write_dataframe(obs_merged2, merge_path2)


def averageTrialsToFinalOutput2(lakename, realization, trials=2, PGRNN=True):
    # for a given lake and realization of randomly chosen observations, compile the experiment results into an averaged prediction
    # @lakename = string nhd id
    # @trials (not implemented yet) = number of trials per experiment setup
    # @PGRNN = True if PGRNN, False if RNN

    realization = str(realization)
    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'
    o_path1 = '../../scripts/manylakes2/outputs' + realization + '/' + lakename + pg + '_output_' + 'trial0.feather'
    o_path2 = '../../scripts/manylakes2/outputs' + realization + '/' + lakename + pg + '_output_' + 'trial1.feather'
    o_path3 = '../../scripts/manylakes2/outputs' + realization + '/' + lakename + pg + '_output_' + 'trial2.feather'
    merge_path = '../../scripts/manylakes2/outputs' + realization + '/' + lakename + pg + '_output_' + 'avg10.feather'
    obs1 = pd.read_feather(o_path1)
    obs2 = pd.read_feather(o_path2)
    obs3 = pd.read_feather(o_path3)
    obs_merged = obs1.copy()
    obs_merged2 = pd.DataFrame().reindex_like(obs1)  #

    obs_merged.values[:, 1:] = (obs1.values[:, 1:] + obs2.values[:, 1:] + obs3.values[:,
                                                                          1:]) / 3  # average the two trials
    feather.write_dataframe(obs_merged, merge_path)

    # obs_merged2.values[:,1:] = (obs1.values[:,1:] + obs2.values[:,1:] + obs3.values[:,1:]) / 3
    # obs_merged2['date'] = pd.to_datetime(df['date'])
    # # obs_merged2.values[:,0] = obs1.values[:,0]
    # feather.write_dataframe(obs_merged2, merge_path2)


def averageTrialsToFinalOutputFullData(lakename, trials=2, PGRNN=True):
    # for a given lake and realization of randomly chosen observations, compile the experiment results into an averaged prediction
    # @lakename = string nhd id
    # @trials (not implemented yet) = number of trials per experiment setup
    # @PGRNN = True if PGRNN, False if RNN

    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'
    o_path1 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial0.feather'
    o_path2 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial1.feather'
    o_path3 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial2.feather'
    o_path4 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial3.feather'
    o_path5 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial4.feather'
    merge_path = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'avgT.feather'
    merge_path2 = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'avg2T.feather'
    obs1 = pd.read_feather(o_path1)
    obs2 = pd.read_feather(o_path2)
    obs3 = pd.read_feather(o_path3)
    obs4 = pd.read_feather(o_path4)
    obs5 = pd.read_feather(o_path5)
    obs_merged = obs1.copy()
    obs_merged2 = pd.DataFrame().reindex_like(obs1)  #

    obs_merged.values[:, 1:] = (obs1.values[:, 1:] + obs2.values[:, 1:] + obs3.values[:, 1:] + obs4.values[:,
                                                                                               1:] + obs5.values[:,
                                                                                                     1:]) / 3  # average the two trials
    feather.write_dataframe(obs_merged, merge_path)

    # obs_merged2.values[:,1:] = (obs1.values[:,1:] + obs2.values[:,1:] + obs3.values[:,1:]) / 3
    # obs_merged2['date'] = pd.to_datetime(df['date'])
    # # obs_merged2.values[:,0] = obs1.values[:,0]
    # feather.write_dataframe(obs_merged2, merge_path2)


def saveFeather(output_npy, label_npy, u_dates, lakename, trial, realization, l1_norm='NA', PGRNN=True, moniker=""):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    # @output_npy = prediction matrix (depths x days)
    # @label_npy = label matrix (depth x days)
    # @u_dates = numpy array of unique dates (np.datetime64 type)
    # @lakename = string nhd id (str)
    # @n_hid = number of best hidden units for experiment (str or int)
    # @realization = realization index from randomization (str or int)
    # @l1_norm (optional) = if l1 norm is used in hyperparameter optimization this is the value
    import feather
    l1_norm = str(l1_norm)
    trial = str(trial)
    realization = str(realization)
    output_df = pd.DataFrame({'date': u_dates})
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = output_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col = pd.DataFrame({'depth_' + str(i): output_npy[i, :]})
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        output_df = pd.concat([output_df, new_col], axis=1)
        label_df = pd.concat([label_df, new_col2], axis=1)
    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'

    o_path = '../../scripts/manylakes2/outputs' + realization + '/' + lakename + pg + '_output_trial' + trial + "_" + moniker + '.feather'
    l_path = '../../scripts/manylakes2/labels/' + lakename + '_label.feather'

    # save em
    exists = os.path.isfile(l_path)

    if not exists:
        feather.write_dataframe(label_df, l_path)
    feather.write_dataframe(output_df, o_path)


def saveFeatherFullData(output_npy, label_npy, u_dates, lakename, trial, PGRNN=True, includeTest=False):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    # @output_npy = prediction matrix (depths x days)
    # @label_npy = label matrix (depth x days)
    # @u_dates = numpy array of unique dates (np.datetime64 type)
    # @lakename = string nhd id (str)
    # @n_hid = number of best hidden units for experiment (str or int)
    # @realization = realization index from randomization (str or int)
    # @l1_norm (optional) = if l1 norm is used in hyperparameter optimization this is the value
    trial = str(trial)
    output_df = pd.DataFrame({'date': u_dates})
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = output_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col = pd.DataFrame({'depth_' + str(i): output_npy[i, :]})
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        output_df = pd.concat([output_df, new_col], axis=1)
        label_df = pd.concat([label_df, new_col2], axis=1)
    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'

    o_path = '../../scripts/manylakes2/outputs_full/' + lakename + pg + '_output_' + 'trial' + trial + '.feather'
    l_path = '../../scripts/manylakes2/labels/' + lakename + '_label.feather'

    # save em
    exists = os.path.isfile(l_path)

    if not exists:
        feather.write_dataframe(label_df, l_path)
    feather.write_dataframe(output_df, o_path)


def saveFeatherFullDataWithEnergy(output_npy, label_npy, energies, u_dates, lakename, targetLake, trial, PGRNN=True,
                                  includeTest=False, moniker=""):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    # @output_npy = prediction matrix (depths x days), numpy array
    # @label_npy = label matrix (depth x days), numpy array
    # @energies = energy every day, numpy array
    # @u_dates = numpy array of unique dates (np.datetime64 type)
    # @lakename = string nhd id (str)
    # @n_hid = number of best hidden units for experiment (str or int)
    # @realization = realization index from randomization (str or int)
    # @l1_norm (optional) = if l1 norm is used in hyperparameter optimization this is the value
    trial = str(trial)
    output_df = pd.DataFrame({'date': u_dates})
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = output_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col = pd.DataFrame({'depth_' + str(i): output_npy[i, :]})
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        output_df = pd.concat([output_df, new_col], axis=1)
        label_df = pd.concat([label_df, new_col2], axis=1)

    if energies is not None:
        new_col = pd.DataFrame({'energy': energies})
        energies_norm = (energies - energies[~np.isnan(energies)].mean()) / energies[~np.isnan(energies)].std()
        new_col2 = pd.DataFrame({'energy_norm': energies_norm})
        output_df = pd.concat([output_df, new_col], axis=1)
        output_df = pd.concat([output_df, new_col2], axis=1)

    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'
    if lakename is None:
        lakename = ""
    o_path = '../../scripts/manylakes2/single_models/' + lakename + pg + 'outputOn' + targetLake + '_trial' + trial + '_' + moniker + '.feather'
    l_path = '../../scripts/manylakes/labels/' + lakename + '_label.feather'

    # save em
    exists = os.path.isfile(l_path)

    if not exists:
        feather.write_dataframe(label_df, l_path)
    print(o_path)
    feather.write_dataframe(output_df, o_path)


def saveFeatherFullDataWithEnergy(output_npy, label_npy, energies, u_dates, lakename, targetLake, trial, PGRNN=True,
                                  includeTest=False, moniker=""):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    # @output_npy = prediction matrix (depths x days), numpy array
    # @label_npy = label matrix (depth x days), numpy array
    # @energies = energy every day, numpy array
    # @u_dates = numpy array of unique dates (np.datetime64 type)
    # @lakename = string nhd id (str)
    # @n_hid = number of best hidden units for experiment (str or int)
    # @realization = realization index from randomization (str or int)
    # @l1_norm (optional) = if l1 norm is used in hyperparameter optimization this is the value
    trial = str(trial)
    output_df = pd.DataFrame({'date': u_dates})
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = output_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col = pd.DataFrame({'depth_' + str(i): output_npy[i, :]})
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        output_df = pd.concat([output_df, new_col], axis=1)
        label_df = pd.concat([label_df, new_col2], axis=1)

    if energies is not None:
        new_col = pd.DataFrame({'energy': energies})
        energies_norm = (energies - energies[~np.isnan(energies)].mean()) / energies[~np.isnan(energies)].std()
        new_col2 = pd.DataFrame({'energy_norm': energies_norm})
        output_df = pd.concat([output_df, new_col], axis=1)
        output_df = pd.concat([output_df, new_col2], axis=1)

    pg = ''
    if PGRNN:
        pg = 'PGRNN'
    else:
        pg = 'RNN'
    if lakename is None:
        lakename = ""
    o_path = '../../scripts/manylakes2/single_models/' + lakename + pg + 'outputOn' + targetLake + '_trial' + trial + '_' + moniker + '.feather'
    l_path = '../../scripts/manylakes/labels/' + lakename + '_label.feather'

    # save em
    exists = os.path.isfile(l_path)

    if not exists:
        feather.write_dataframe(label_df, l_path)
    print(o_path)
    feather.write_dataframe(output_df, o_path)


def saveTemperatureMatrix(output_npy, label_npy, u_dates, lakename, targetLake, save_path="", label_path=""):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    # @output_npy = prediction matrix (depths x days), numpy array
    # @label_npy = label matrix (depth x days), numpy array
    # @u_dates = numpy array of unique dates (np.datetime64 type)
    # @lakename = string nhd id (str)
    output_df = pd.DataFrame({'date': u_dates})
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = output_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col = pd.DataFrame({'depth_' + str(i): output_npy[i, :]})
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        output_df = pd.concat([output_df, new_col], axis=1)
        label_df = pd.concat([label_df, new_col2], axis=1)

    o_path = save_path
    l_path = label_path

    # save em
    exists = os.path.isfile(l_path)

    if not exists:
        feather.write_dataframe(label_df, l_path)
    feather.write_dataframe(output_df, o_path)


def makeLabels(label_npy, u_dates, lakename, full=True):
    # convert predictions/labels numpy arrays into pandas dataframe and save as feather
    import feather
    label_df = pd.DataFrame({'date': u_dates})
    n_test_dates = u_dates.shape[0]
    n_depths = label_npy.shape[0]
    for i in range(0, n_depths):
        data = np.empty((n_test_dates))
        data[:] = np.nan
        new_col2 = pd.DataFrame({'depth_' + str(i): label_npy[i, :]})
        label_df = pd.concat([label_df, new_col2], axis=1)

    l_path = '../../scripts/manylakes/labels/' + lakename + '_test_label.feather'

    if full:
        l_path = '../../scripts/manylakes/labels/' + lakename + '_full_label.feather'

    # save em
    # exists = os.path.isfile(l_path)
    print("writing to ", l_path)
    feather.write_dataframe(label_df, l_path)
    sys.exit()
