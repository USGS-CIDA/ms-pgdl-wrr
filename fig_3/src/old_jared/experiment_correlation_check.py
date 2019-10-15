from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import random
import math
import sys
import os
sys.path.append('../../data')
sys.path.append('../../models')
sys.path.append('/home/invyz/workspace/Research/lake_monitoring/src/data')
from rw_data import readMatData
# from data_operations import calculatePhysicalLossDensityDepth
from pytorch_data_operations import buildLakeDataForRNNPretrain, calculate_ec_loss_manylakes, transformTempToDensity
from pytorch_model_operations import saveModel
import pytorch_data_operations

#multiple dataloader wrapping?
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

####################################################3
#  pretrain script, takes lakename as required command line argument
###################################################33

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)
# data_parallel = True

lakename = sys.argv[1]



#define loss and optimizer
hid_low = 7
hid_high = 17
hidden_unit_arr = np.arange(hid_low,hid_high)
trial_per_hid = 2 # trials per hidden unit tuning trial
# n_trials = 5 #final number of trials with optimal hidden value
err_per_trial = np.empty((hidden_unit_arr.shape[0], trial_per_hid))
#############################################################
#training loop
####################################################################
for n_hidden in range(hid_low,hid_high):
    print("n hid = ", n_hidden)
    #define LSTM object instance
    #define LSTM model class



    for trial in range(trial_per_hid):
        print("trial ", trial)
        #####################3
        #params
        ###########################33
        n_ep = 600  #number of epochs
        first_save_epoch = 0
        patience = 100

        #ow
        seq_length = 200 #how long of sequences to use in model
        begin_loss_ind = 100 #index in sequence where we begin to calculate error or predict
        n_features = 8  #number of physical drivers
        win_shift = 50 #how much to slide the window on training set each time
        save = True 
        data_dir = "../../data/processed/WRR_69Lake/"+lakename+"/"


        ###############################
        # data preprocess
        ##################################
        #create train and test sets
        (trn_data, all_data, all_phys_data, all_dates,
        hypsography) = buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features,
                                           win_shift= win_shift, begin_loss_ind=begin_loss_ind)
        n_depths = torch.unique(all_data[:,:,0]).size()[0]


        ####################
        #model params
        ########################

        batch_size =600
        yhat_batch_size = n_depths*1
        grad_clip = 1.0 #how much to clip the gradient 2-norm in training
        lambda1 = 0.0000#magnitude hyperparameter of l1 loss
        ec_lambda = 0.1 #magnitude hyperparameter of ec loss
        ec_threshold = 36 #anything above this far off of energy budget closing is penalized
        dc_lambda = 0. #magnitude hyperparameter of depth-density constraint (dc) loss

        #Dataset classes
        class TemperatureTrainDataset(Dataset):
            #training dataset class, allows Dataloader to load both input/target
            def __init__(self, trn_data):
                # depth_data = depth_trn
                self.len = trn_data.shape[0]
                # assert data.shape[0] ==trn_data depth_data.shape[0]
                self.x_data = trn_data[:,:,:-1].float()
                # self.x_depth = depth_data.float()
                self.y_data = trn_data[:,:,-1].float()

            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]

            def __len__(self):
                return self.len

        class TotalModelOutputDataset(Dataset):
            #dataset for unsupervised input(in this case all the data)
            def __init__(self):
                #data of all model output, and corresponding unstandardized physical quantities
                #needed to calculate physical loss
                self.len = all_data.shape[0]
                self.data = all_data[:,:,:-1].float()
                self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
                self.phys = all_phys_data[:,:,:].float()
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
        # print(depth_areas)
        if use_gpu:
            depth_areas = depth_areas.cuda()

        #format total y-hat data for loading
        total_data = TotalModelOutputDataset()
        n_batches = math.floor(trn_data.size()[0] / batch_size)

        assert yhat_batch_size == n_depths

        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)



        #method to calculate l1 norm of model
        def calculate_l1_loss(model):
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


        class myLSTM_Net(nn.Module):
            def __init__(self, input_size, hidden_size, batch_size):
                super(myLSTM_Net, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) 
                self.out = nn.Linear(hidden_size, 1)
                self.hidden = self.init_hidden()

            def init_hidden(self, batch_size=0):
                # initialize both hidden layers
                if batch_size == 0:
                    batch_size = self.batch_size
                ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                        xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
                if use_gpu:
                    item0 = ret[0].cuda(non_blocking=True)
                    item1 = ret[1].cuda(non_blocking=True)
                    ret = (item0,item1)
                return ret
            
            def forward(self, x, hidden):
                self.lstm.flatten_parameters()
                x = x.float()
                x, hidden = self.lstm(x, self.hidden)
                self.hidden = hidden
                x = self.out(x)
                return x, hidden


        lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)
            #tell model to use GPU if needed
        if use_gpu:
            print("putting on gpu")
            lstm_net = lstm_net.cuda(0)

        mse_criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_net.parameters())#, weight_decay=0.01)


        ind = n_hidden-hid_low
        save_path = "../../../models/WRR_69Lake/"+lakename+"/pretrain_experiment_nhid"+str(n_hidden)+"_trial"+str(trial)

        min_loss = 99999
        min_mse_tsterr = None
        ep_min_mse = -1
        epoch_since_best = 0

        manualSeed = [random.randint(1, 99999999) for i in range(n_ep)]
        for epoch in range(n_ep):
            print("pretrain epoch: ", epoch)
            # random.seed(opt.manualSeed)
            torch.manual_seed(manualSeed[epoch])
            if use_gpu:
                torch.cuda.manual_seed_all(manualSeed[epoch])
            # print("epoch ", epoch+1)
            running_loss = 0.0

            #reload loader for shuffle
            #batch samplers used to draw samples in dataloaders
            batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

            batch_sampler_all = pytorch_data_operations.RandomContiguousBatchSampler(all_data.size()[0], seq_length, yhat_batch_size, n_batches)
            alldataloader = DataLoader(total_data, batch_sampler=batch_sampler_all, pin_memory=True)
            trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
            multi_loader = pytorch_data_operations.MultiLoader([trainloader, alldataloader])


            #zero the parameter gradients
            optimizer.zero_grad()
            avg_loss = 0
            batches_done = 0
            for i, batches in enumerate(multi_loader):
                # print("batch ", i)
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
                if(use_gpu):
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                #forward  prop
                lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
                h_state = None
                inputs = inputs.float()
                outputs, h_state = lstm_net(inputs, h_state)
                outputs = outputs.view(outputs.size()[0],-1)

                loss_outputs = outputs[:,begin_loss_ind:]
                loss_targets = targets[:,begin_loss_ind:]
                #unsupervised output

                h_state = None
                lstm_net.hidden = lstm_net.init_hidden(batch_size = yhat_batch_size)
                # use_gpu = False
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
                    # depth_area_percs = depth_area_percs.cuda()

                    # lstm_net = lstm_net.cuda()
                unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)
                # unsup_outputs = unsup_outputs.cuda()

                #calculate unsupervised loss

                # dc_unsup_loss = torch.tensor(0).float()
                # if use_gpu:
                #     dc_unsup_loss = dc_unsup_loss.cuda()

                # if dc_lambda > 0:
                #     dc_unsup_loss = calculate_dc_loss(unsup_outputs, n_depths, use_gpu)

                if ec_lambda > 0: #if we are calculating energy loss
                    unsup_loss = calculate_ec_loss_manylakes(unsup_inputs[:,begin_loss_ind:,:],
                                               unsup_outputs[:,begin_loss_ind:,:],
                                               unsup_phys_data[:,begin_loss_ind:,:],
                                               unsup_labels[:,begin_loss_ind:],
                                               unsup_dates[:,begin_loss_ind:],                                        
                                               depth_areas,
                                               n_depths,
                                               ec_threshold,
                                               use_gpu, 
                                               combine_days=1)
                

                #calculate losses
                reg1_loss = 0
                if lambda1 > 0:
                    reg1_loss = calculate_l1_loss(lstm_net)

                loss = mse_criterion(loss_outputs, loss_targets) + lambda1*reg1_loss + ec_lambda*unsup_loss


                mse_loss = mse_criterion(loss_outputs, loss_targets) 
                loss = mse_loss+ lambda1*reg1_loss + ec_lambda*unsup_loss
                mse_loss = mse_criterion(loss_outputs, loss_targets) 
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

                #print statistics
                running_loss += loss.item()
                if i % 3 == 2:
                    #print('[%d, %5d] loss: %.3f' %
                    #      (epoch + 1, i + 1, running_loss / 3))
                    running_loss = 0.0
            if avg_loss < min_loss:
                if epoch+1 > first_save_epoch:
                        #save model if best
                        if save:
                            print("saved at", save_path)
                            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

                        epoch_since_best = 0
                min_loss = avg_loss
                min_mse_tsterr = mse_loss
                ep_min_mse = epoch +1
                epoch_since_best += 1
            # print("Epoch with min loss: ", ep_min_mse, " -> loss=", min_loss, "with mse=", min_mse_tsterr)

            if epoch_since_best == patience:
                print("pretraining finished")
                continue
                # sys.exit()        
         

        from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune
        ####################################################3
        # fine tune
        ###################################################33


        #####################3
        #params
        ###########################33
        n_ep = 10000  #number of epochs
        first_save_epoch = 1
        patience = 1000
        epoch_since_best = 0

        #how
        # lakename2 = 'Southern_lake'
        win_shift = 25 #how much to slide the window on training set each time
        data_dir = "../../data/processed/WRR_69Lake/"+lakename+"/"
        pretrain_path = "../../../models/WRR_69Lake/"+lakename+"/pretrain_experiment_nhid"+str(n_hidden)+"_trial"+str(trial)
        save_path = "../../../models/WRR_69Lake/"+lakename+"/finetune_"+str(n_hidden)+"_"+str(trial)
        ###############################
        # data preprocess
        ##################################
        #create train and test sets
        (trn_data, tst_data, all_data, all_phys_data, all_dates,
        hypsography) = buildLakeDataForRNN_manylakes_finetune(lakename, data_dir, seq_length, n_features,
                                           win_shift= win_shift, begin_loss_ind=begin_loss_ind, correlation_check=True)


        n_depths = torch.unique(all_data[:,:,0]).size()[0]

        ####################
        #model params
        ########################
        # n_hidden = 8 #number of hidden units in LSTM
        # batch_size = 600
        # yhat_batch_size = n_depths*1
        # grad_clip = 1.0 #how much to clip the gradient 2-norm in training
        # lambda1 = 0.00 #magnitude hyperparameter of l1 loss
        # ec_lambda = 0.0 #magnitude hyperparameter of ec loss
        # ec_threshold = 36 #anything above this far off of energy budget closing is penalized


        #Dataset classes
        class TemperatureTrainDataset(Dataset):
            #training dataset class, allows Dataloader to load both input/target
            def __init__(self, trn_data):
                # depth_data = depth_trn
                self.len = trn_data.shape[0]
                self.x_data = trn_data[:,:,:-1].float()
                self.y_data = trn_data[:,:,-1].float()

            def __getitem__(self, index):
                return self.x_data[index], self.y_data[index]

            def __len__(self):
                return self.len

        class TotalModelOutputDataset(Dataset):
            #dataset for unsupervised input(in this case all the data)
            def __init__(self):
                #data of all model output, and corresponding unstandardized physical quantities
                #needed to calculate physical loss
                self.len = all_data.shape[0]
                self.data = all_data[:,:,:-1].float()
                self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
                self.phys = all_phys_data.float()
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
        yhat_batch_size = n_depths

        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

        # del mat
        # print("trn_input: ", trn_input.size(), "\n trn_target: ", trn_target.size())
        # print("train data size, ", trn_data.size())
        # print("all data size, ", all_data.size())


        #load val/test data into enumerator based on batch size
        testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)


        #define LSTM model class
        class myLSTM_Net(nn.Module):
            def __init__(self, input_size, hidden_size, batch_size):
                super(myLSTM_Net, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True) #batch_first=True?
                self.out = nn.Linear(hidden_size, 1) #1?
                self.hidden = self.init_hidden()

            def init_hidden(self, batch_size=0):
                # initialize both hidden layers
                if batch_size == 0:
                    batch_size = self.batch_size
                ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                        xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
                # print("hidden layer initialized: ", ret)
                # if data_parallel: #TODO??
                #     ret = (xavier_normal_(torch.empty(1, math.ceil(self.batch_size/2), self.hidden_size/2)),
                #         xavier_normal_(torch.empty(1, math.floorself.batch_size, math.floor(self.hidden_size/2))))
                if use_gpu:
                    item0 = ret[0].cuda(non_blocking=True)
                    item1 = ret[1].cuda(non_blocking=True)
                    ret = (item0,item1)
                return ret
            
            def forward(self, x, hidden):
                # print("X size is {}".format(x.size()))
                x = x.float()
                x, hidden = self.lstm(x, self.hidden)
                self.hidden = hidden
                x = self.out(x)
                return x, hidden

        #method to calculate l1 norm of model
        def calculate_l1_loss(model):
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


        lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)

        pretrain_dict = torch.load(pretrain_path)['state_dict']
        model_dict = lstm_net.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        lstm_net.load_state_dict(pretrain_dict)

        #tell model to use GPU if needed
        if use_gpu:
            lstm_net = lstm_net.cuda()

        # print("model created")

        #define loss and optimizer
        mse_criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_net.parameters())#, weight_decay=0.01)

        #training loop

        min_mse = 99999
        min_mse_tsterr = None
        ep_min_mse = -1
        manualSeed = [random.randint(1, 99999999) for i in range(n_ep)]
        for epoch in range(n_ep):
            print("train epoch: ", epoch)
            # random.seed(opt.manualSeed)
            torch.manual_seed(manualSeed[epoch])
            if use_gpu:
                torch.cuda.manual_seed_all(manualSeed[epoch])
            # print("epoch ", epoch+1)
            running_loss = 0.0

            #reload loader for shuffle
            #batch samplers used to draw samples in dataloaders
            batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
            batch_sampler_all = pytorch_data_operations.RandomContiguousBatchSampler(all_data.size()[0], seq_length, yhat_batch_size, n_batches)
            alldataloader = DataLoader(total_data, batch_sampler=batch_sampler_all, pin_memory=True)
            trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
            multi_loader = pytorch_data_operations.MultiLoader([trainloader, alldataloader])


            #zero the parameter gradients
            optimizer.zero_grad()
            for i, batches in enumerate(multi_loader):
                # print("batch ", i)
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


                # del batches

                #cuda commands
                if(use_gpu):
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                #forward  prop
                lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
                h_state = None
                outputs, h_state = lstm_net(inputs, h_state)
                outputs = outputs.view(outputs.size()[0],-1)

                #unsupervised output

                h_state = None
                lstm_net.hidden = lstm_net.init_hidden(batch_size = yhat_batch_size)
                # use_gpu = False
                unsup_loss = torch.tensor(0).float()
                if use_gpu:
                    unsup_inputs = unsup_inputs.cuda()
                    unsup_phys_data = unsup_phys_data.cuda()
                    unsup_labels = unsup_labels.cuda()
                    depth_areas = depth_areas.cuda()
                    unsup_dates = unsup_dates.cuda()
                    unsup_loss = unsup_loss.cuda()
                    # depth_area_percs = depth_area_percs.cuda()

                    # lstm_net = lstm_net.cuda()
                unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)
                # unsup_outputs = unsup_outputs.cuda()

                #calculate unsupervised loss

                if ec_lambda > 0:
                    unsup_loss = calculate_ec_loss_manylakes(unsup_inputs[:,begin_loss_ind:,:],
                                               unsup_outputs[:,begin_loss_ind:,:],
                                               unsup_phys_data[:,begin_loss_ind:,:],
                                               unsup_labels[:,begin_loss_ind:],
                                               unsup_dates[:,begin_loss_ind:],                                        
                                               depth_areas,
                                               n_depths,
                                               ec_threshold,
                                               use_gpu, 
                                               combine_days=1)
                # print(unsup_loss)
                # print("unsup loss ", unsup_loss)
                #calculate losses
                reg1_loss = 0
                if lambda1 > 0:
                    reg1_loss = calculate_l1_loss(lstm_net)


                loss_outputs = outputs[:,begin_loss_ind:]
                loss_targets = targets[:,begin_loss_ind:]
                if use_gpu:
                    loss_outputs = loss_outputs.cuda()
                    loss_targets = loss_targets.cuda()

                loss_indices = np.where(~np.isnan(loss_targets))
                if ~np.isfinite(loss_targets).any():
                    print("loss targets should not be nan shouldnt happen")
                    sys.exit()
                loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices]) + lambda1*reg1_loss + ec_lambda*unsup_loss
                #backward
                loss.backward(retain_graph=False)
                if grad_clip > 0:
                    clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

                #optimize
                optimizer.step()

                #zero the parameter gradients
                optimizer.zero_grad()

            #compute test loss
            with torch.no_grad():
                avg_mse = 0
                for i, data in enumerate(testloader, 0):
                    inputs = data[:,:,:n_features].float()
                    targets = data[:,:,-1].float()
                    targets = targets[:, begin_loss_ind:]
                    # print(torch.unique(inputs[:,0,0]).size()[0])
                    # print(n_depths)
                    # assert torch.unique(inputs[:,0,0]).size()[0] == n_depths
                    if use_gpu:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    h_state = None
                    lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
                    pred, h_state = lstm_net(inputs, h_state)
                    pred = pred.view(pred.size()[0],-1)
                    pred = pred[:, begin_loss_ind:]
                    loss_indices = np.where(~np.isnan(targets))

                    mse = mse_criterion(pred[loss_indices], targets[loss_indices])
                    avg_mse += mse
                avg_mse = avg_mse / len(testloader)
                if avg_mse < min_mse:
                    if epoch+1 > first_save_epoch:
                        #save model if best
                        if save:
                            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

                        epoch_since_best = 0
                    min_mse = avg_mse
                    ep_min_mse = epoch +1
                epoch_since_best += 1

            if epoch_since_best == patience:
                print("rmse=", np.sqrt(min_mse))
                err_per_trial[ind, trial] = min_mse
                break
                # print("Epoch with min mse: ", ep_min_mse, " -> ", min_mse)
       

hidden_unit_arr = np.arange(hid_low, hid_high)
min_ind = np.argmin(err_per_trial[:,0]+err_per_trial[:,1])
best_hid = hidden_unit_arr[min_ind]
# print(err_per_hid)
print("best hidden unit =", best_hid)
print(err_per_trial)
print(err_per_trial[best_hid,:].mean())


np.savetxt("PGRNNrmse"+lakename+".csv", err_per_trial[best_hid,:].mean(), delimiter=",")            # print("Epoch with min mse: ", ep_min_mse, " -> ", min_mse)
   