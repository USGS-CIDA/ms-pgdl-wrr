# https://github.com/jdwillard19/lake_modeling/blob/e024238617f00b0a34a18698bd1aff5484d87005/src/scripts/manylakes2/pgrnn_figure3.py

"""This script was run to create the data for Figure 3 in the Read et al 2019 WRR submission (Jared 09/2019)"""
from __future__ import print_function
import datetime
from datetime import date
import random
import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
import pandas as pd
sys.path.append('../../data')
sys.path.append('../../models')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')
import pytorch_data_operations
from pytorch_data_operations import buildLakeDataForRNNPretrain, \
                                    calculate_ec_loss_manylakes
from pytorch_model_operations import saveModel
from io_operations import averageTrialsToFinalOutputFullData, saveFeatherFullData


####################################################3
#  takes lakename as required command line argument
###################################################33

#enable/disable cuda
use_gpu = True
torch.backends.cudnn.benchmark = True #optimize cuda
torch.set_printoptions(precision=10) #set sig figs for printing

#collect command line args
lakename = sys.argv[1]

#print time script ran
current_dt = datetime.datetime.now()
print(str(current_dt))

# trials to average at end
n_trials = 5
err_per_trial = np.empty((n_trials))


### debug tools
debug_train = False
verbose = False

n_hidden = 20 #fixed

#####################3
#params
###########################33
n_ep = 400  #number of epochs

#parameters
seq_length = 352 #how long of sequences to use in model
begin_loss_ind = 50#index in sequence where we begin to calculate error or predict
n_features = 8  #number of physical drivers
win_shift = 176 #how much to slide the window on training set each time
save = True
data_dir = "../../data/processed/WRR_69Lake/"+lakename+"/"


###############################
# data preprocess
##################################
#create train and test sets
(trn_data, all_data, all_phys_data, all_dates,
 hypsography) = buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features,
                                            win_shift=win_shift, begin_loss_ind=begin_loss_ind,
                                            excludeTest=False)
for trial in range(n_trials): #training loop
    print("trial ", trial)

    n_depths = torch.unique(all_data[:, :, 0]).size()[0]

    ####################
    #model params
    ########################

    batch_size = trn_data.size()[0]
    yhat_batch_size = n_depths*1 #how many sequences to calculate unsupervised loss every epoch
    grad_clip = 1.0 #how much to clip the gradient 2-norm in training
    lambda1 = 0.0000#magnitude hyperparameter of l1 loss
    ec_lambda = 0.1 #magnitude hyperparameter of ec loss
    ec_threshold = 36 #anything above this far off of energy budget closing is penalized
    dc_lambda = 0. #magnitude hyperparameter of depth-density constraint (dc) loss

    #Dataset classes
    class TemperatureTrainDataset(Dataset):
        """training dataset class, allows Dataloader to load both input/target"""
        def __init__(self, full_data_arg):
            self.len = full_data_arg.shape[0]
            self.x_data = full_data_arg[:, :, :-1].float()
            self.y_data = full_data_arg[:, :, -1].float()

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len

    class TotalModelOutputDataset(Dataset):
        """dataset for unsupervised input(in this case all the data"""
        def __init__(self):
            #data of all model output, and corresponding unstandardized physical quantities
            #needed to calculate physical loss
            self.len = all_data.shape[0]
            self.data = all_data[:, :, :-1].float()
            self.label = all_data[:, :, -1].float() #DO NOT USE IN MODEL
            self.phys = all_phys_data[:, :, :].float()
            helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
            dates = helper(all_dates)
            self.dates = dates

        def __getitem__(self, index):
            return self.data[index], self.phys[index], self.dates[index], self.label[index]

        def __len__(self):
            return self.len

    #format training data for loading
    train_data = TemperatureTrainDataset(trn_data)

    #get depth area percent data
    depth_areas = torch.from_numpy(hypsography).float()[0]

    if use_gpu:
        depth_areas = depth_areas.cuda()

    #format total y-hat data for loading
    total_data = TotalModelOutputDataset()
    n_batches = math.floor(trn_data.size()[0] / batch_size)

    assert yhat_batch_size == n_depths

    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


    def calculate_l1_loss(model):
        """method to calculate l1 norm of model"""
        def l1_loss(x):
            return torch.abs(x).sum()

        to_regularize = []
        # for name, p in model.named_parameters():
        for name, p in model.named_parameters():
            if 'bias' in name:
                continue
            else:
                #take absolute value of weights and sum
                to_regularize.append(p.view(-1))
        l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
        l1_loss_val = l1_loss(torch.cat(to_regularize))
        return l1_loss_val


    class my_lstm_net(nn.Module):
        """lstm class"""
        def __init__(self, input_size, hidden_size, batch_size):
            """initialization"""
            super(my_lstm_net, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
            self.out = nn.Linear(hidden_size, 1)
            self.hidden = self.init_hidden()

        def init_hidden(self, batch_size=0):
            """initialize both hidden layers"""
            if batch_size == 0:
                batch_size = self.batch_size
            ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                   xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
            if use_gpu:
                item0 = ret[0].cuda(non_blocking=True)
                item1 = ret[1].cuda(non_blocking=True)
                ret = (item0, item1)
            return ret

        def forward(self, x, hidden):
            """forward prop"""
            self.lstm.flatten_parameters()
            x = x.float()
            x, hidden = self.lstm(x, self.hidden)
            self.hidden = hidden
            x = self.out(x)
            return x, hidden

    #instantiate LSTM, move to GPU
    lstm_net = my_lstm_net(n_features, n_hidden, batch_size)
    if use_gpu:
        lstm_net = lstm_net.cuda(0)

    #define training loss function and optimizer
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=.005)#, weight_decay=0.01)

    #path to save pre-trained model
    save_path = "../../../models/WRR_69Lake/"+lakename+"/pretrain_experiment_trial"+str(trial)

    #training parameters
    min_loss = 99999
    min_mse_tsterr = None
    ep_min_mse = -1
    manualSeed = [random.randint(1, 99999999) for i in range(n_ep)]

    for epoch in range(n_ep):
        if verbose:
            print("pretrain epoch: ", epoch)
        torch.manual_seed(manualSeed[epoch])
        if use_gpu:
            torch.cuda.manual_seed_all(manualSeed[epoch])

        #reload loader for shuffle
        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
        batch_sampler_all = pytorch_data_operations.RandomContiguousBatchSampler(all_data.size()[0],
                                                                                 seq_length,
                                                                                 yhat_batch_size,
                                                                                 n_batches)
        alldataloader = DataLoader(total_data, batch_sampler=batch_sampler_all, pin_memory=True)
        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader([trainloader, alldataloader])


        #zero the parameter gradients
        optimizer.zero_grad()


        avg_loss = 0
        batches_done = 0
        for i, batches in enumerate(multi_loader):
            #load data
            inputs = None
            targets = None
            depths = None
            unsup_inputs = None
            unsup_phys_data = None
            unsup_depths = None
            unsup_dates = None
            unsup_labels = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets = b

                if j == 1:
                    unsup_inputs, unsup_phys_data, unsup_dates, unsup_labels = b



            #cuda commands
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            inputs = inputs.float()
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0], -1)

            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]

            #unsupervised output
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=yhat_batch_size)
            unsup_loss = torch.tensor(0).float()
            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()
                unsup_inputs = unsup_inputs.cuda()
                unsup_phys_data = unsup_phys_data.cuda()
                unsup_labels = unsup_labels.cuda()
                depth_areas = depth_areas.cuda()
                unsup_dates = unsup_dates.cuda()
                unsup_loss = unsup_loss.cuda()

            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            #calculate unsupervised loss
            if ec_lambda > 0: #if we are calculating energy loss
                unsup_loss = calculate_ec_loss_manylakes(unsup_inputs[:, begin_loss_ind:, :],
                                                         unsup_outputs[:, begin_loss_ind:, :],
                                                         unsup_phys_data[:, begin_loss_ind:, :],
                                                         unsup_labels[:, begin_loss_ind:],
                                                         unsup_dates[:, begin_loss_ind:],
                                                         depth_areas,
                                                         n_depths,
                                                         ec_threshold,
                                                         use_gpu,
                                                         combine_days=1)
            #calculate losses
            reg1_loss = 0
            if lambda1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)

            loss = mse_criterion(loss_outputs, loss_targets) + \
                                 lambda1*reg1_loss +  \
                                 ec_lambda*unsup_loss


            avg_loss += loss

            batches_done += 1
            #backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            #optimize
            optimizer.step()

            #zero the parameter gradients
            optimizer.zero_grad()


    print("pre-train finished")
    saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, \
                                        parseMatricesFromSeqs



    #####################################################################################
    ####################################################3
    # fine tune
    ###################################################33
    ##########################################################################################33

    #####################3
    #params
    ###########################33
    n_ep = 400  #number of epochs
    win_shift = 176 #how much to slide the window on training set each time
    data_dir = "../../data/processed/WRR_69Lake/"+lakename+"/"
    pretrain_path = "../../../models/WRR_69Lake/"+lakename+"/pretrain_experiment_trial"+str(trial)
    save_path = "../../../models/WRR_69Lake/"+lakename+"/pgrnn_finetune_trial"+str(trial)

    ###############################
    # data preprocess
    ##################################
    #create train and test sets
    (trn_data, trn_dates, tst_data, tst_dates, unique_tst_dates, all_data, all_phys_data,
     all_dates, hypsography) = buildLakeDataForRNN_manylakes_finetune2(lakename, \
                                                            data_dir, \
                                                            seq_length, n_features, \
                                                            win_shift=win_shift, \
                                                            begin_loss_ind=begin_loss_ind, \
                                                            latter_third_test=True, \
                                                            outputFullTestMatrix=True, \
                                                            sparseTen=False, \
                                                            realization='none', \
                                                            allTestSeq=True)



    batch_size = trn_data.size()[0]
    n_test_dates = unique_tst_dates.shape[0]
    n_depths = torch.unique(all_data[:, :, 0]).size()[0]
    u_depths = np.unique(tst_data[:, 0, 0])

    #format training data for loading
    train_data = TemperatureTrainDataset(trn_data)

    #get depth area percent data
    depth_areas = torch.from_numpy(hypsography).float()[0]
    if use_gpu:
        depth_areas = depth_areas.cuda()

    #format total y-hat data for loading
    total_data = TotalModelOutputDataset()
    n_batches = math.floor(trn_data.size()[0] / batch_size)
    yhat_batch_size = n_depths

    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)



    #load val/test data into enumerator based on batch size
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], \
                                             shuffle=False, pin_memory=True)

    #instantiate LSTM model
    lstm_net = my_lstm_net(n_features, n_hidden, batch_size)

    #load pre-trianed LSTM
    pretrain_dict = torch.load(pretrain_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #tell model to use GPU if needed
    if use_gpu:
        lstm_net = lstm_net.cuda()

    #define loss and optimizer
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=.005)#, weight_decay=0.01)

    #training params
    min_mse = 99999
    min_mse_tsterr = None
    ep_min_mse = -1
    best_pred_mat = np.empty(())
    manualSeed = [random.randint(1, 99999999) for i in range(n_ep)]

    #matrix output/label data structures
    output_npy = np.empty((n_depths, n_test_dates))
    label_npy = np.empty((n_depths, n_test_dates))
    output_npy[:] = np.nan
    label_npy[:] = np.nan

    for epoch in range(n_ep): #fine tune training loop
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            torch.cuda.manual_seed_all(manualSeed[epoch])

        #reload loader for shuffle
        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
        batch_sampler_all = pytorch_data_operations.RandomContiguousBatchSampler(all_data.size()[0],
                                                                                 seq_length,
                                                                                 yhat_batch_size,
                                                                                 n_batches)
        alldataloader = DataLoader(total_data, batch_sampler=batch_sampler_all, pin_memory=True)
        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader([trainloader, alldataloader])


        #zero the parameter gradients
        optimizer.zero_grad()
        lstm_net.train(True)
        avg_loss = 0
        avg_unsup_loss = 0
        batches_done = 0
        for i, batches in enumerate(multi_loader):
            #load data
            inputs = None
            targets = None
            depths = None
            unsup_inputs = None
            unsup_phys_data = None
            unsup_depths = None
            unsup_dates = None
            unsup_labels = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets = b

                if j == 1:
                    unsup_inputs, unsup_phys_data, unsup_dates, unsup_labels = b

            #cuda commands
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0], -1)

            #unsupervised output
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=yhat_batch_size)
            unsup_loss = torch.tensor(0).float()
            if use_gpu:
                unsup_inputs = unsup_inputs.cuda()
                unsup_phys_data = unsup_phys_data.cuda()
                unsup_labels = unsup_labels.cuda()
                depth_areas = depth_areas.cuda()
                unsup_dates = unsup_dates.cuda()
                unsup_loss = unsup_loss.cuda()

            #get unsupervised outputs
            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            #calculate unsupervised loss
            if ec_lambda > 0:
                unsup_loss = calculate_ec_loss_manylakes(unsup_inputs[:, begin_loss_ind:, :],
                                                         unsup_outputs[:, begin_loss_ind:, :],
                                                         unsup_phys_data[:, begin_loss_ind:, :],
                                                         unsup_labels[:, begin_loss_ind:],
                                                         unsup_dates[:, begin_loss_ind:],
                                                         depth_areas,
                                                         n_depths,
                                                         ec_threshold,
                                                         use_gpu,
                                                         combine_days=1)

            #calculate losses
            reg1_loss = 0
            if lambda1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)


            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]
            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()

            #get indices to calculate loss
            loss_indices = np.where(~np.isnan(loss_targets))
            if ~np.isfinite(loss_targets).any():
                print("loss targets should not be nan shouldnt happen")
                sys.exit()
            loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices]) + \
                                 lambda1*reg1_loss + ec_lambda*unsup_loss
            #backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            #optimize
            optimizer.step()

            #zero the parameter gradients
            optimizer.zero_grad()
            avg_loss += loss
            avg_unsup_loss += unsup_loss
            batches_done += 1


    lstm_net.eval() #set LSTM to evaluation mode
    with torch.no_grad(): #disable gradient
        avg_mse = 0
        for i, data in enumerate(testloader, 0):
            #this loop is dated, there is now only one item in testloader

            #parse data into inputs and targets
            inputs = data[:, :, :n_features].float()
            targets = data[:, :, -1].float()
            targets = targets[:, begin_loss_ind:]
            tmp_dates = tst_dates[:, begin_loss_ind:]
            depths = inputs[:, :, 0]

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #run model
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            pred, h_state = lstm_net(inputs, h_state)
            pred = pred.view(pred.size()[0], -1)
            pred = pred[:, begin_loss_ind:]

            #calculate error
            loss_indices = np.where(~np.isnan(targets))
            inputs = inputs[:, begin_loss_ind:, :]
            depths = depths[:, begin_loss_ind:]
            mse = mse_criterion(pred[loss_indices], targets[loss_indices])
            avg_mse += mse

            #fill in data structs to save model outputs
            (output_npy, label_npy) = parseMatricesFromSeqs(pred, targets, depths, \
                                                            tmp_dates, n_depths, \
                                                            n_test_dates, u_depths, \
                                                            unique_tst_dates)
            #save model
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in "+ str(epoch) + " epochs")
            print("finished trial", trial)
            print("rmse=", np.sqrt(avg_mse))
            loss_output = output_npy[~np.isnan(label_npy)]
            loss_label = label_npy[~np.isnan(label_npy)]
            mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
            print("Total rmse=", mat_rmse)
            saveFeatherFullData(output_npy, label_npy, unique_tst_dates, lakename, trial)
            err_per_trial[trial] = mat_rmse



#print results
print(err_per_trial)
print(err_per_trial[:].mean())
averageTrialsToFinalOutputFullData(lakename, trials=5, PGRNN=True)
