import torch
import numpy as np
import pandas as pd
import math
import sys
import phys_operations
import datetime
from datetime import date
import os


def getFullLakeData(lakename, n_features):
    #load data created in preprocess.py based on lakename
    feat_mat_raw = np.load("../../../data/processed/"+lakename+"/features.npy")
    feat_mat = np.load("../../../data/processed/"+lakename+"/processed_features.npy")
    dates = np.load("../../../data/processed/"+lakename+"/dates.npy")
    print("DATES: ", dates.size)
    Y_mat = np.load("../../../data/processed/"+lakename+"/labels.npy")
    return (feat_mat, feat_mat_raw, Y_mat)

def buildLakeDataForRNN(lakename, seq_length, n_features, train_split=0.2, val_split = 0.1, win_shift= 1, sparseness=0, flip_trn_test=False, begin_loss_ind = 102, specific_train_year=None):
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@train_split = percentage of data to be used as training(started from beginning of data unless flip specified)
        #@val_split = percentage of data to be used as validation, started at end of train data
        #@win_shift = days to move in the sliding window for the training set
        #@sparseness = specify to be less than 1 if you want to randomly hide a percentage of the train/val data
        #@flip_trn_test = set to True to use end of data as train and val
    #load data created in preprocess.py based on lakename
    my_path = os.path.abspath(os.path.dirname(__file__))

    feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features.npy"))
    feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features.npy"))
    Y_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/labels.npy"))
    diag = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/diag.npy"))
    dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    if specific_train_year is not None:
      #move specific year to front

        #remove first year, not necessary if data starts at Jan 1
        print("first partial year not used")
        first_partial_year_ind = np.where(years == years.min())[0]
        feat_mat_raw = np.delete(feat_mat_raw, first_partial_year_ind,1)
        feat_mat = np.delete(feat_mat, first_partial_year_ind,1)
        Y_mat = np.delete(Y_mat, first_partial_year_ind,1)
        diag = np.delete(diag, first_partial_year_ind,1)
        dates = np.delete(dates, first_partial_year_ind,0)

        year_ind = np.where(years == specific_train_year)[0] #train year
        first_year_ind = np.arange(year_ind.size) #first year in array

        #get data to move to front
        new_train_feat_raw = feat_mat_raw[:,year_ind,:]
        new_train_feat = feat_mat[:,year_ind,:]
        new_train_label = Y_mat[:,year_ind]
        new_train_diag = diag[:,year_ind,:]
        new_train_dates = dates[year_ind]

        #move front data to training year
        feat_mat_raw[:,year_ind,:] = feat_mat_raw[:,first_year_ind,:]
        feat_mat[:,year_ind,:] = feat_mat[:,first_year_ind,:]
        Y_mat[:,year_ind] = Y_mat[:,first_year_ind]
        diag[:,year_ind,:] = diag[:,first_year_ind,:]
        dates[year_ind] = dates[first_year_ind]

        #move training data to front
        feat_mat_raw[:,first_year_ind,:] = new_train_feat_raw
        feat_mat[:,first_year_ind,:] = new_train_feat
        Y_mat[:,first_year_ind] = new_train_label
        diag[:,first_year_ind,:] = new_train_diag
        dates[first_year_ind] = new_train_dates




    # print("DATES: ", dates.size)

    # print(feat_mat_raw[0:50,100,:])
    # print(feat_mat[0:50,100,:])
    # sys.exit()
    assert np.isfinite(feat_mat).any(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).any(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    n_depths = feat_mat.shape[0]
    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == Y_mat.shape[0]
    assert feat_mat.shape[1] == Y_mat.shape[1]
    assert feat_mat.shape[1] == feat_mat_raw.shape[1]
    assert feat_mat.shape[0] == diag.shape[0]
    val_win_per_seq = 0
    tst_win_per_seq = 1
    if val_split > 0:
        val_win_per_seq = math.floor(seq_length / begin_loss_ind)
        tst_win_per_seq = val_win_per_seq

    win_shift_val = begin_loss_ind
    win_shift_tst = begin_loss_ind
    depth_values = feat_mat_raw[:, 0, 1]
    assert np.unique(depth_values).size == n_depths
    udates = dates
    n_dates = feat_mat.shape[1]
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = math.floor(train_split*(seq_per_depth))
    val_seq_per_depth = np.maximum(math.floor(val_split*seq_per_depth-1),0)
    test_seq_per_depth = seq_per_depth - train_seq_per_depth - val_seq_per_depth 
    win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_train_seq_no_window = train_seq_per_depth * n_depths
    n_val_seq = val_seq_per_depth * n_depths * val_win_per_seq
    n_test_seq = (test_seq_per_depth) * n_depths * tst_win_per_seq
    print("n_test_seq: ",n_test_seq)

    n_all_seq = n_train_seq_no_window + n_val_seq + n_test_seq

    #make sparse mask
    if sparseness < 0 or sparseness > 1:
        print("Invalid sparseness value, must be between zero and one")
    sparse_mask = np.ones_like(feat_mat[:,:,0])
    n_data_to_remove_per_day = int(np.round(feat_mat.shape[0] * sparseness))
    def removeData(a):
        a[np.random.choice(n_depths, size=n_data_to_remove_per_day, replace=False)] = 0
        return a
    sparse_mask = np.apply_along_axis(removeData, 0, sparse_mask)


    #build train and test sets, add all data for physical loss
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+2)) #features + sparse bool + label
    X_val = np.array([])
    if n_val_seq > 0:
        X_val = np.empty(shape=(n_val_seq, seq_length, n_features+1)) 
        X_val[:] = np.nan

    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1)) #include date now
    tst_phys = np.empty(shape=(n_test_seq, seq_length,10))
    tst_dates = np.empty(shape=(n_test_seq, seq_length), dtype='datetime64[s]')
    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    X_phys = np.empty(shape=(n_all_seq, seq_length, 10)) #short wave, long wave, modeled temp, depth

    X_trn[:] = np.nan
    X_tst[:] = np.nan
    X_all[:] = np.nan
    X_phys[:] = np.nan
    tst_phys[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    ts_seq_ind = 0
    val_seq_ind = 0
    all_seq_ind = 0
    s_skipped = 0
    #build datasets
    print("before train data")
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        for d in range(0, n_depths):
            #first do total model data
            X_all[all_seq_ind, :, :-1] = feat_mat[d,start_index:end_index,:] #feat
            all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
            X_all[all_seq_ind,:,-1] = Y_mat[d,start_index:end_index] #label
            X_phys[all_seq_ind, :, :-3] = feat_mat_raw[d, start_index:end_index,0:7]
            X_phys[all_seq_ind, :, -3:] = diag[d, start_index:end_index,:]
            # X_phys[all_seq_ind, :, 7] = feat_mat_raw[d, start_index:end_index,2]  
            all_seq_ind += 1   
        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length
            if win_end_ind > n_dates:
                continue
            for d in range(0,n_depths):
                X_trn[tr_seq_ind, :, :-2] = feat_mat[d,win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind, :, -2] = sparse_mask[d,win_start_ind:win_end_ind]
                X_trn[tr_seq_ind,:,-1] = Y_mat[d,win_start_ind:win_end_ind]
                tr_seq_ind += 1
    #assert data was constructed correctly
    assert tr_seq_ind == n_train_seq, \
        "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)
    print("train set created")
    if n_val_seq > 0:
        #now val data(maybe bug in this specification of end of range?)
        for s in range(train_seq_per_depth+1,train_seq_per_depth+val_seq_per_depth):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                if end_index > n_dates:
                    continue
                for d in range(0,n_depths):
                    X_all[all_seq_ind, :, :-1] = feat_mat[d,start_index:end_index,:] #feat
                    all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
                    X_all[all_seq_ind,:,-1] = Y_mat[d,start_index:end_index] #label
                    X_phys[all_seq_ind, :, :-3] = feat_mat_raw[d, start_index:end_index,0:7]
                    X_phys[all_seq_ind, :, -3:] = diag[d, start_index:end_index,:]
                    all_seq_ind += 1   
                    #val_seq_ind += 1
                for w in range(0, val_win_per_seq):
                    win_start_ind = start_index+w*win_shift_val
                    win_end_ind = win_start_ind + seq_length
                    if win_end_ind > n_dates:
                        continue
                    for d in range(0, n_depths):
                        X_val[val_seq_ind, :, :-1] = feat_mat[d,win_start_ind:win_end_ind,:]
                        X_val[val_seq_ind,:,-1] = Y_mat[d,win_start_ind:win_end_ind]
                        val_seq_ind += 1
    #assert data was constructed correctly  
    assert val_seq_ind == n_val_seq, \
        "incorrect number of val seq estimated {} vs actual{}".format(n_val_seq, val_seq_ind)   
    print("val set created")
    if n_test_seq != 0:
        #now test data(maybe bug in this specification of end of range?)
        for s in range(train_seq_per_depth+val_seq_per_depth-1,seq_per_depth-1):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                if end_index > n_dates:
                    continue
                for d in range(0,n_depths):
                    #X_tst[ts_seq_ind,:,:-1] = feat_mat[d,start_index:end_index,:]
                    #tst_dates[ts_seq_ind, :] = dates[start_index:end_index] #dates
                    #X_tst[ts_seq_ind,:,-1] = Y_mat[d,start_index:end_index]
                    #tst_phys[ts_seq_ind, :, :-3] = feat_mat_raw[d, start_index:end_index,0:7]
                    #tst_phys[ts_seq_ind, :, -3:] = diag[d, start_index:end_index,:]
                    X_all[all_seq_ind, :, :-1] = feat_mat[d,start_index:end_index,:] #feat
                    all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
                    X_all[all_seq_ind,:,-1] = Y_mat[d,start_index:end_index] #label
                    X_phys[all_seq_ind, :, :-3] = feat_mat_raw[d, start_index:end_index,0:7]
                    X_phys[all_seq_ind, :, -3:] = diag[d, start_index:end_index,:]                    # tst_dates[ts_seq_ind,:,0] = udates[start_index:end_index,0]

                    #ts_seq_ind += 1
                    all_seq_ind += 1
                for w in range(0, tst_win_per_seq):
                    win_start_ind = start_index+w*win_shift_tst
                    win_end_ind = win_start_ind + seq_length
                    if win_end_ind > n_dates:
                        continue
                    for d in range(0, n_depths):
                        X_tst[ts_seq_ind, :, :-1] = feat_mat[d,win_start_ind:win_end_ind,:]
                        X_tst[ts_seq_ind,:,-1] = Y_mat[d,win_start_ind:win_end_ind]
                        tst_dates[ts_seq_ind, :] = dates[win_start_ind:win_end_ind] #dates
                        tst_phys[ts_seq_ind, :, :-3] = feat_mat_raw[d, win_start_ind:win_end_ind,0:7]
                        tst_phys[ts_seq_ind, :, -3:] = diag[d, win_start_ind:win_end_ind,:]

                        ts_seq_ind += 1

    #assert data was constructed correctly
    assert ts_seq_ind == n_test_seq, \
        "incorrect number of tst seq estimated {} vs actual{}".format(n_test_seq, ts_seq_ind)      
    print("test set created")
    #debug statements
    # print("trn seq: ", tr_seq_ind)
    # print("val seq: ", val_seq_ind)
    # print("tst seq: ", ts_seq_ind)

    # print("all seq: ", all_seq_ind)
    #X_trnval = np.vstack((X_trn, X_val)) #train data + validation data for final model construction

    assert X_tst.shape[0] == tst_phys.shape[0] 
    # assert X_tst.shape[0] == tst_dates.shape[0]

    depths = np.unique(tst_phys[:,:,1])
    hyps = getHypsography(lakename, depths)

    ## make train and val sparse by sparseness factor, build mask
    trn_mask = np.random.choice([0, 1], (X_trn.shape[0], X_trn.shape[1]), p=[sparseness, 1-sparseness])

    val_mask = np.array([])
    if n_val_seq > 0:
        val_mask = np.random.choice([0, 1], (X_val.shape[0], X_val.shape[1]), p=[sparseness, 1-sparseness])
    assert np.isfinite(X_trn).all(), "X_trn has nan at" + str(np.argwhere(np.isfinite(X_trn)))
    # assert np.isfinite(X_val).any(), "X_val has nan" + str(np.argwhere(np.isfinite(X_val)))
    # assert np.isfinite(X_tst).any(), "X_tst has nan"
    assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
    assert np.isfinite(X_phys).all(), "X_phys has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), torch.from_numpy(trn_mask), torch.from_numpy(X_val), torch.from_numpy(val_mask),
                torch.from_numpy(X_tst), 
                torch.from_numpy(tst_phys), tst_dates,
                torch.from_numpy(X_all), torch.from_numpy(X_phys), all_dates,
                hyps
                )

def buildLakeDataForRNN_manylakes_finetune(lakename, data_dir, seq_length, n_features, win_shift= 1, begin_loss_ind = 102, n_trn_obs=-1, test_seq_per_depth=1,correlation_check=False):
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)
    #load data created in preprocess.py based on lakename
    my_path = os.path.abspath(os.path.dirname(__file__))

    feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/features.npy"))
    feat_mat = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/processed_features.npy"))
    tst = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/test.npy"))
    trn = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/train.npy"))
    if correlation_check:
        tst = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/test_b.npy"))
        trn = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/train_b.npy"))
    dates = np.load(os.path.join(my_path, "../../data/processed/WRR_69Lake/"+lakename+"/dates.npy"))
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    print("loaded data files")
    assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    n_depths = feat_mat.shape[0]
    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == tst.shape[0]
    assert feat_mat.shape[0] == trn.shape[0]
    assert feat_mat.shape[1] == tst.shape[1]
    assert feat_mat.shape[1] == trn.shape[1]
    assert feat_mat.shape[1] == feat_mat_raw.shape[1]
    win_shift_tst = begin_loss_ind
    depth_values = feat_mat_raw[:, 0, 0]
    assert np.unique(depth_values).size == n_depths
    udates = dates
    n_dates = feat_mat.shape[1]
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = seq_per_depth
    test_seq_per_depth = seq_per_depth
    win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
    tst_win_per_seq = 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_train_seq_no_window = train_seq_per_depth * n_depths
    n_test_seq = (test_seq_per_depth) * n_depths * tst_win_per_seq

    n_all_seq = n_train_seq_no_window 


    #build train and test sets, add all data for physical loss
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1)) 
    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    X_phys = np.empty(shape=(n_all_seq, seq_length, n_features+1)) #non-normalized features + ice cover flag

    X_trn[:] = np.nan
    X_tst[:] = np.nan
    X_all[:] = np.nan
    X_phys[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    ts_seq_ind = 0
    all_seq_ind = 0
    #build datasets
    del_all_seq = 0
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        if end_index > n_dates:
            n_train_seq -= win_per_seq*n_depths
            n_all_seq -= n_depths
            del_all_seq += 1
            X_all = np.delete(X_all, np.arange(X_all.shape[0],X_all.shape[0]-n_depths,-1), axis=0)
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
            continue
        for d in range(0, n_depths):
            #first do total model data
            X_all[all_seq_ind, :, :-1] = feat_mat[d,start_index:end_index,:] #feat
            all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
            X_all[all_seq_ind,:,-1] = np.nan #no label
            X_phys[all_seq_ind, :, :] = feat_mat_raw[d, start_index:end_index,:]
            all_seq_ind += 1   
        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length
            if win_end_ind > n_dates:
                n_train_seq -= 1
                X_trn = np.delete(X_trn, -1, axis=0)
                continue
            for d in range(0,n_depths):
                X_trn[tr_seq_ind, :, :-1] = feat_mat[d,win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind,:,-1] = trn[d,win_start_ind:win_end_ind]
                tr_seq_ind += 1
    #assert data was constructed correctly
    if tr_seq_ind != n_train_seq:
        print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
        extra = n_train_seq - tr_seq_ind
        n_train_seq -= extra
        X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
    assert tr_seq_ind == n_train_seq, \
     "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)



    if n_test_seq != 0:
        #now test data(maybe bug in this specification of end of range?)
        for s in range(test_seq_per_depth):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                if end_index > n_dates:
                    n_test_seq -= tst_win_per_seq*n_depths
                    X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - tst_win_per_seq*n_depths,-1), axis=0)
                    continue
                for w in range(0, tst_win_per_seq):
                    win_start_ind = start_index+w*win_shift_tst
                    win_end_ind = win_start_ind + seq_length
                    if win_end_ind > n_dates:
                        n_test_seq -= 1
                        X_tst = np.delete(X_tst, -1, axis=0)
                        continue
                    for d in range(0, n_depths):
                        X_tst[ts_seq_ind, :, :-1] = feat_mat[d,win_start_ind:win_end_ind,:]
                        X_tst[ts_seq_ind,:,-1] = tst[d,win_start_ind:win_end_ind]
                        ts_seq_ind += 1

    #assert data was constructed correctly
    assert ts_seq_ind == n_test_seq, \
        "incorrect number of tst seq estimated {} vs actual{}".format(n_test_seq, ts_seq_ind)      

    #remove sequences with no labels
    tr_seq_removed = 0
    trn_del_ind = np.array([], dtype=np.int32)
    ts_seq_removed = 0
    tst_del_ind = np.array([], dtype=np.int32)

    print("n train seq", n_train_seq)
    print("n test seq", n_test_seq)
    for i in range(n_train_seq):
        if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
            continue
        else:
            # print(X_trn[i,:,-1])
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)

    for i in range(n_test_seq):
        if np.isfinite(X_tst[i,begin_loss_ind:,-1]).any():
            continue
        else:
            tst_del_ind = np.append(tst_del_ind, i)
            ts_seq_removed += 1

    # print("X_trn shape ", X_trn.shape)
    # print("trn_del_ind ", trn_del_ind)
    X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
    X_tst_tmp = np.delete(X_tst, tst_del_ind, axis=0)
    X_trn = X_trn_tmp
    X_tst = X_tst_tmp

    # print("old ", X_trn.shape[0], ", new ", new_trn.shape[0])
    # print(tr_seq_removed, " trn seq deleted out of ", n_train_seq)
    # print(ts_seq_removed, " tst seq deleted out of ", n_test_seq)
    # assert X_trn.shape[0] - new_trn.shape[0] == tr_seq_removed
    # assert X_tst.shape[0] - new_tst.shape[0] == ts_seq_removed



    hyps_dir = data_dir + "geometry"
    hyps = getHypsographyManyLakes(hyps_dir, lakename, depth_values)
    # print(X_all[:,:,:-1])
    # print(del_all_seq)
    assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
    assert np.isfinite(X_phys).all(), "X_phys has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), torch.from_numpy(X_tst), torch.from_numpy(X_all), 
            torch.from_numpy(X_phys), all_dates, hyps)

def buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features, win_shift= 1, begin_loss_ind = 102):
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@train_split = percentage of data to be used as training(started from beginning of data unless flip specified)
        #@val_split = percentage of data to be used as validation, started at end of train data
        #@win_shift = days to move in the sliding window for the training set
        #@sparseness = specify to be less than 1 if you want to randomly hide a percentage of the train/val data
        #@flip_trn_test = set to True to use end of data as train and val
    #load data created in preprocess.py based on lakename
    my_path = os.path.abspath(os.path.dirname(__file__))

    feat_mat_raw = np.load(os.path.join(my_path, data_dir +"features.npy"))
    feat_mat = np.load(os.path.join(my_path, data_dir + "processed_features.npy"))
    Y_mat = np.load(os.path.join(my_path, data_dir +"glm.npy"))

    # diag = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/diag.npy"))
    dates = np.load(os.path.join(my_path, data_dir +"dates.npy"))
    years = dates.astype('datetime64[Y]').astype(int) + 1970
 

    # print("DATES: ", dates.size)

    # print(feat_mat_raw[0:50,100,:])
    # print(feat_mat[0:50,100,:])
    # sys.exit()
    assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    assert np.isfinite(Y_mat).all(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    n_depths = feat_mat.shape[0]
    print(n_depths)
    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == Y_mat.shape[0]
    assert feat_mat.shape[1] == Y_mat.shape[1]
    assert feat_mat.shape[1] == feat_mat_raw.shape[1]
    depth_values = feat_mat_raw[:, 0, 0]
    assert np.unique(depth_values).size == n_depths
    udates = dates
    n_dates = feat_mat.shape[1]
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = seq_per_depth
    # val_seq_per_depth = np.maximum(math.floor(val_split*seq_per_depth-1),0)
    # test_seq_per_depth = seq_per_depth - train_seq_per_depth - val_seq_per_depth 
    win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * n_depths * win_per_seq
    n_train_seq_no_window = train_seq_per_depth * n_depths
    n_all_seq = n_train_seq_no_window



    #build train and test sets, add all data for physical loss
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
    # X_val = np.array([])

    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
    X_phys = np.empty(shape=(n_all_seq, seq_length, n_features+1)) #short wave, long wave, modeled temp, depth

    X_trn[:] = np.nan
    X_all[:] = np.nan
    X_phys[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    all_seq_ind = 0
    s_skipped = 0
    #build datasets
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        for d in range(0, n_depths):
            #first do total model data
            X_all[all_seq_ind, :, :-1] = feat_mat[d,start_index:end_index,:] #feat
            all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
            X_all[all_seq_ind,:,-1] = Y_mat[d,start_index:end_index] #label
            X_phys[all_seq_ind, :, :] = feat_mat_raw[d, start_index:end_index,:]
            all_seq_ind += 1   
        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length
            if win_end_ind > n_dates:
                continue
            for d in range(0,n_depths):
                X_trn[tr_seq_ind, :, :-1] = feat_mat[d,win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind,:,-1] = Y_mat[d,win_start_ind:win_end_ind]
                tr_seq_ind += 1
    #assert data was constructed correctly
    if tr_seq_ind != n_train_seq:
        print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
        extra = n_train_seq - tr_seq_ind
        n_train_seq -= extra
        X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
    assert tr_seq_ind == n_train_seq, \
        "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

    print("train set created")
    

    # depths = np.unique(tst_phys[:,:,1])
    hyps_dir = data_dir + "geometry"
    hyps = getHypsographyManyLakes(hyps_dir, lakename, depth_values)

    ## make train and val sparse by sparseness factor, build mask
    assert np.isfinite(X_trn).any(), "X_trn has nan at" + str(np.argwhere(np.isfinite(X_trn)))
    assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
    assert np.isfinite(X_phys).any(), "X_phys has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), torch.from_numpy(X_all), torch.from_numpy(X_phys), all_dates, hyps)






def plot_ec_loss(lakename,all_data, all_phys_data, all_dates,hypsography, n_depths, depth_areas, combine_days=1):
    depth_areas = torch.from_numpy(depth_areas[0]).double()
    use_gpu = False
    import numpy as np
    errors = np.zeros(shape=(366))
    n_days = np.zeros(shape=(366))
    # print(all_data.size())
    # print(all_dates.shape)
    assert all_data.size()[0] == all_phys_data.size()[0]
    assert all_phys_data.size()[0] == all_dates.shape[0]
    n_sets = math.floor(all_data.size()[0] / n_depths)
    x = all_data[:,:,:-1]
    y = all_data[:,:,-1]
    densities = transformTempToDensity(y, use_gpu)
    print(n_sets, " sets of depths")
    for s in range(n_sets):
        print("set ", s)
        #indices
        start_index = (s)*n_depths
        end_index = (s+1)*n_depths
        # diff_vec = torch.empty((x.size()[1]))
        data_list = [pd.to_datetime(pd.Series(x), format="%Y%m%d") for x in all_dates[start_index,:]]
        # print(data_list)
        doy = pd.DataFrame([x.apply(lambda x: x.timetuple().tm_yday) for x in data_list]).values
        # doy = [datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in all_dates[start_index,:]]
        # print(doy)
        lake_energies = calculate_lake_energy(y[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)
        #calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        lake_energy_fluxes = calculate_energy_fluxes(all_phys_data[start_index,:,:], y[start_index,:], combine_days)
        ### can use this to plot energy delta and flux over time to see if they line up

        diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
        for t in range(diff_vec.size()[0]):
            # print("doy[t], ", doy[t])
            # print("t, ", t)
            errors[doy[t]-1] += diff_vec[t]
            n_days[doy[t]-1] += 1
        # print("errors: ", errors)
        # print("n days: ", n_days)

    avg_err = errors / n_days    
    import matplotlib
    import matplotlib.pyplot as plt
    t = np.arange(366)
    fig, ax = plt.subplots()
    ax.plot(t, avg_err, label="error")
    # ax.plot(t, lake_energy_fluxes.numpy(), label="surface fluxes")
    # ax.plot(t, lake_energy_fluxes.numpy() - lake_energy_deltas.numpy(), label="DIFF")

    ax.set(xlabel="doy", ylabel='W/m^2 diff',title='Average Absolute Difference between sum of fluxes and lake energy change for GLM on '+lakename)
    plt.legend()
    plt.show()


def get_energy_diag(inputs, outputs, phys, labels, dates, depth_areas, n_depths, use_gpu, combine_days=1):
    import numpy as np
    n_sets = int(inputs.size()[0] / n_depths) #sets of depths in batch
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]

    # outputs = labels

    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  

    #calculate lake energy for each timestep
    lake_energies = calculate_lake_energy(outputs[:,:], densities[:,:], depth_areas)
    #calculate energy change in each timestep
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
    lake_energy_deltas = lake_energy_deltas[1:]
    #calculate sum of energy flux into or out of the lake at each timestep
    # print("dates ", dates[0,1:6])
    lake_energy_fluxes = calculate_energy_fluxes(phys[0,:,:], outputs[0,:], combine_days)
    ### can use this to plot energy delta and flux over time to see if they line up

    
    # mendota og ice guesstimate
    # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

    # #actual ice 
    # diff_vec = diff_vec[np.where((phys[:,9] == 0))[0]]

    # # #compute difference to be used as penalty
    # diff_per_set[i] = diff_vec.mean()
    return (lake_energy_deltas.numpy(), lake_energy_fluxes.numpy())

def calculate_ec_loss_manylakes(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days=1):
    import numpy as np
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
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]
    # outputs = labels  
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  
        #loop through sets of n_depths

 

    #calculate lake energy for each timestep

    lake_energies = calculate_lake_energy(outputs[:,:], densities[:,:], depth_areas)
    #calculate energy change in each timestep
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
    lake_energy_deltas = lake_energy_deltas[1:]
    #calculate sum of energy flux into or out of the lake at each timestep
    # print("dates ", dates[0,1:6])
    lake_energy_fluxes = calculate_energy_fluxes_manylakes(phys[0,:,:], outputs[0,:], combine_days)
    ### can use this to plot energy delta and flux over time to see if they line up
    # doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
    # doy = doy[1:-1]


    diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
    diff_vec = diff_vec[np.where((phys[0,1:-1,-1] == 0))[0]] #only over no-ice period

    if use_gpu:
        diff_vec = diff_vec.cuda()
    #actual ice
    # diff_vec = diff_vec[np.where((phys[1:(n_depths-diff_vec.size()[0]-1),9] == 0))[0]]
    # #compute difference to be used as penalty
    if diff_vec.size() == torch.Size([0]):
        return 0
    else:
        res = torch.clamp(diff_vec.mean() - ec_threshold, min=0)  
        return res

def calculate_dc_loss(outputs, n_depths, use_gpu):
    #calculates depth-density consistency loss
    #parameters:
        #@outputs: labels = temperature predictions, organized as depth (rows) by date (cols)
        #@n_depths: number of depths
        #@use_gpu: gpu flag

    assert outputs.size()[0] == n_depths

    densities = transformTempToDensity(outputs, use_gpu)

    # We could simply count the number of times that a shallower depth (densities[:-1])
    # has a higher density than the next depth below (densities[1:])
    # num_violations = (densities[:-1] - densities[1:] > 0).sum()

    # But instead, let's use sum(sum(ReLU)) of the density violations,
    # per Karpatne et al. 2018 (https://arxiv.org/pdf/1710.11431.pdf) eq 3.14
    sum_violations = (densities[:-1] - densities[1:]).clamp(min=0).sum()

    return sum_violations


def calculate_ec_loss(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days=1):
    import numpy as np
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
        #@combine_days: how many days to look back to see if energy is conserved (obsolete)
    #*********************************************************************************

    n_sets = math.floor(inputs.size()[0] / n_depths)#sets of depths in batch
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]

    
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  
    diff_per_set = torch.empty(n_sets) 
    for i in range(n_sets):
        #loop through sets of n_depths

        #indices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths


        #assert have all depths
        # assert torch.unique(inputs[:,0,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,100,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,200,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,0,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,100,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,200,1]).size()[0] == n_depths


        #calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)
        #calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index,:,:], outputs[start_index,:], combine_days)
        ### can use this to plot energy delta and flux over time to see if they line up
        doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
        doy = doy[1:-1]
        diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
        
        # mendota og ice guesstimate
        # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

        #actual ice
        diff_vec = diff_vec[np.where((phys[0,1:-1,9] == 0))[0]]
        # #compute difference to be used as penalty
        if diff_vec.size() == torch.Size([0]):
            diff_per_set[i] = 0
        else:
            diff_per_set[i] = diff_vec.mean()
    if use_gpu:
        diff_per_set = diff_per_set.cuda()
    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    print(diff_per_set.mean())
    return diff_per_set.mean()

def calculate_lake_energy(temps, densities, depth_areas):
    #calculate the total energy of the lake for every timestep
    #sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    #then multiply by the specific heat of water 
    dz = 0.5 #thickness for each layer, hardcoded for now
    cw = 4186 #specific heat of water
    energy = torch.empty_like(temps[0,:])
    n_depths = depth_areas.size()[0]
    depth_areas = depth_areas.view(n_depths,1).expand(n_depths, temps.size()[1])
    energy = torch.sum(depth_areas*temps*densities*0.5*cw,0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    #given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400 #seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1])/(time*surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas





def calculate_energy_fluxes(phys, surf_temps, combine_days):
    # print("surface_depth = ", phys[0:5,1])
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400 #seconds per day
    surface_area = 39865825 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_arr = phys[:-1,3] + (phys[1:,3]-phys[:-1,3])/2
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,4] 
    air_temp2 = phys[1:,4]
    rel_hum = phys[:-1,5]
    rel_hum2 = phys[1:,5]
    ws = phys[:-1, 6]
    ws2 = phys[1:,6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    #test
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])


    return fluxes

def calculate_energy_fluxes_manylakes(phys, surf_temps, combine_days):
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400 #seconds per day
    surface_area = 39865825 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,1] + (phys[1:,1]-phys[:-1,1])/2
    R_lw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,3] 
    air_temp2 = phys[1:,3]
    rel_hum = phys[:-1,4]
    rel_hum2 = phys[1:,4]
    ws = phys[:-1, 5]
    ws2 = phys[1:,5]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    #test
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])


    return fluxes

def getHypsographyManyLakes(path, lakename, depths):
    my_path = os.path.abspath(os.path.dirname(__file__))
    depth_areas = pd.read_csv(os.path.join(my_path, path), header=0, index_col=0, squeeze=True).to_dict()
    tmp = {}
    total_area = 0
    for key, val in depth_areas.items():
        total_area += val

    for depth in depths:
        #find depth with area that is closest
        depth_w_area = min(list(depth_areas.keys()), key=lambda x:abs(x-depth))
        tmp[depth] = depth_areas[depth_w_area]
    depth_areas = {}

    for k, v in tmp.items():
        total_area += v

    for k, v in tmp.items():
        depth_areas[k] = tmp[k] 

    return np.sort(-np.array([list(depth_areas.values())]))*-1


def getHypsography(lakename, depths, debug=False):
    my_path = os.path.abspath(os.path.dirname(__file__))
    depth_areas = pd.read_csv(os.path.join(my_path, '../../data/raw/'+lakename+'/'+lakename+'_hypsography.csv'), header=0, index_col=0, squeeze=True).to_dict()
    tmp = {}
    total_area = 0
    for key, val in depth_areas.items():
        total_area += val

    for depth in depths:
        #find depth with area that is closest
        depth_w_area = min(list(depth_areas.keys()), key=lambda x:abs(x-depth))
        tmp[depth] = depth_areas[depth_w_area]
    depth_areas = {}

    for k, v in tmp.items():
        total_area += v

    for k, v in tmp.items():
        depth_areas[k] = tmp[k] 

    return np.sort(-np.array([list(depth_areas.values())]))*-1


class ContiguousBatchSampler(object):
    def __init__(self, batch_size, n_batches):
        self.sampler = torch.randperm(n_batches)
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size

class RandomContiguousBatchSampler(object):
    def __init__(self, n_dates, seq_length, batch_size, n_batches):
        # print("n_dates", n_dates)
        # print("seq_len, ", seq_length)
        # print("batch_size ", batch_size)
        # print("n_batches ", n_batches)
        #todo, finish
        self.sampler = torch.randint_like(torch.empty(n_batches), low=0, high=math.floor((n_dates-seq_length)/batch_size))
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size

def transformTempToDensity(temp, use_gpu):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    densities = torch.empty_like(temp)
    if use_gpu:
        temp = temp.cuda()
        densities = densities.cuda()
    # return densities
    # print(densities.size()
    # print(temp.size())
    densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863,2))/(508929.2*(temp[:]+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))
    # print("DENSITIES")
    # for i in range(10):
    #     print(densities[i,i])

    return densities


#Iterator through multiple dataloaders
class MyIter(object):
  """An iterator."""
  def __init__(self, my_loader):
    self.my_loader = my_loader
    self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
    # print("init", self.loader_iters)

  def __iter__(self):
    return self

  def __next__(self):
    # When the shortest loader (the one with minimum number of batches)
    # terminates, this iterator will terminates.
    # The `StopIteration` raised inside that shortest loader's `__next__`
    # method will in turn gets out of this `__next__` method.
    # print("next",     print(self.loader_iters))
    batches = [loader_iter.next() for loader_iter in self.loader_iters]
    return self.my_loader.combine_batch(batches)

  # Python 2 compatibility
  next = __next__

  def __len__(self):
    return len(self.my_loader)

#wrapper class for multiple dataloaders
class MultiLoader(object):
  """This class wraps several pytorch DataLoader objects, allowing each time 
  taking a batch from each of them and then combining these several batches 
  into one. This class mimics the `for batch in loader:` interface of 
  pytorch `DataLoader`.
  Args: 
    loaders: a list or tuple of pytorch DataLoader objects
  """
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MyIter(self)

  def __len__(self):
    return min([len(loader) for loader in self.loaders])

  # Customize the behavior of combining batches here.
  def combine_batch(self, batches):
    return batches

