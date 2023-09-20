from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from datetime import datetime
from model import M3X1

import torch.nn.functional as F
import torch.nn as nn
import hparams as hp
import helper as h
import numpy as np
import torch as t
import shutil
import sys
import os

TRAINING_LABEL = "wav2vec_w10"

## Start
log_path = TRAINING_LABEL + "/"
log_file = log_path + "log.txt"

if os.path.exists(log_path):
    sys.exit("This training label already exists :( \n\"{}\""
             .format(TRAINING_LABEL))
else:
    os.mkdir(log_path)
    os.mkdir(log_path + "/code")
    os.mkdir(log_path + "/results")

for f in os.listdir("./"):
    if f.endswith(".py"):
        shutil.copyfile(f, log_path + "/code/" + f)

h.print_log('\n\tTraining "{}" started at: {} \n'.format(TRAINING_LABEL,
            datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)
h.print_log('GPU: {}'.format(t.cuda.get_device_name()), log_file)
h.print_log('Properties: {}'.format(t.cuda.get_device_properties("cuda")),
            log_file)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


## Load data
dataset_path = "../dataset_endlich/Dataset_Wav2vec20_10s/"

trainset, validset, testset = h.split_dataset(dataset_path = dataset_path,
                                              log_file = log_file,
                                              results_file = log_path + '/results',
                                              bad_ops = ['OP_022', 'OP_028', 
                                               'OP_037', 'OP_040', 'OP_036'],
                                              random_seed = hp.random_seed,
                                              test_split = hp.test_split)


## Compute class weights of phases
label_count = t.zeros(9)
for op in trainset:
    op_label = t.load(dataset_path + op + "/labels")
    label_count += t.bincount(op_label.int().squeeze())
h.print_log("\tLabel Count\t\t: {}".format(label_count), log_file)

cls_num_list = label_count.cpu().clone().detach().numpy()
cls_num_list[-1] = 9999 # manipulate transition phase for less importance 

beta = 0.9999
effective_num = 1.0 - np.power(beta, cls_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
h.print_log("\tPhase Weights\t: {}".format(per_cls_weights), log_file)


## Model
model = M3X1().to(device)
h.print_log('\tNumber of Parameters\t: {:,}'.format(sum(p.numel() 
    for p in model.parameters() if p.requires_grad)), log_file)


## Loss function
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m = 0.5, 
                 weight = None, s = 35):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = t.cuda.FloatTensor(m_list)
        self.m_list = m_list

        assert s > 0
        self.s = s
       
        self.device = device

        self.weight = t.from_numpy(weight).to(device)

    def forward(self, x, target):
        index = t.zeros_like(x, dtype=t.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(t.cuda.FloatTensor)
        batch_m = t.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = t.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target, weight = self.weight, 
                               ignore_index = 8)

loss_func = LDAMLoss(cls_num_list = cls_num_list, device = device, 
                     weight = per_cls_weights)


## Set up optimizer 
optimizer = t.optim.Adam(model.parameters(), 
    lr = hp.learning_rate, weight_decay = hp.weight_decay)


## Dataset class
class OPDataset(Dataset):
    def __init__(self, op):
        self.data = self.get_data(op)

    def __getitem__(self,index):
        x = self.data[0][index].clone().detach().float().to(device)
        p = self.data[1][index].clone().detach().float().to(device)
        a = self.data[2][index].clone().detach().float().to(device)
        g = self.data[3][index].clone().detach().float().to(device)
        y = self.data[4][index].clone().detach().long().to(device)
        return [x, p, a, g, y]
    
    def __len__(self):
        return self.data[4].shape[0]

    def get_data(self, op):
        x_ray = t.load(dataset_path + op + "/x_ray")
        p_mic = t.load(dataset_path + op + "/physician_mic")
        a_mic = t.load(dataset_path + op + "/assistant_mic")
        g_mic = t.load(dataset_path + op + "/ambient_mic")
        label = t.load(dataset_path + op + "/labels")
        return [x_ray, p_mic, a_mic, g_mic, label]


## Train
patience = 0
best_loss = 1E10 # just a large number
early_callback = False

train_loss = t.zeros(hp.epochs, device = device)
valid_loss = t.zeros(hp.epochs, device = device)
valid_metrics = np.zeros((hp.epochs, len(validset), 5)) # [acc, recall, precision, f1, jaccard]


# iter 
for e in range(hp.epochs):
    if early_callback == False:
        h.print_log("\nEpoch\t: {}/{}".format(e + 1, hp.epochs), log_file)
        
        # train
        for i_op, op in enumerate(trainset):
            print("\t\tEpoch progress: {:.2f} % - {}".format(i_op / 
                        len(trainset) * 100, op), end = '\r')
            
            op_dataset = OPDataset(op)
            op_dataloader = DataLoader(dataset = op_dataset, 
                                    batch_size = hp.batch_size,
                                    shuffle = False)
            
            y_prev = t.zeros((1, hp.compute_dim), device = device) #(1, 128)
            for x, p, a, g, y in op_dataloader:
                optimizer.zero_grad()

                y_hat = model(x, p, a, g, y_prev.clone().detach())

                error_step = 0
                for s in range(hp.num_stages):
                    error_step += loss_func(y_hat[:,:,s].squeeze(dim = -1), 
                                            y.squeeze(dim = -1))
                error_step.backward()

                train_loss[e] += error_step

                optimizer.step()

                y_prev[0, -min(hp.compute_dim, x.shape[0]):] = t.argmax(y_hat[:,:,-1], 
                                            dim = 1)[-min(hp.compute_dim, x.shape[0]):]
                
        
        train_loss[e] /= len(trainset)
        h.print_log("\t\tt_loss\t:{:.5f}\n".format(train_loss[e]), log_file)

        # validate
        for i_op, op in enumerate(validset):
            print("\t\tEpoch progress: {:.2f} % - {}".format(i_op / 
                        len(validset) * 100, op), end = '\r')
            
            model.eval()

            op_dataset = OPDataset(op)
            op_dataloader = DataLoader(dataset = op_dataset, 
                                    batch_size = hp.batch_size,
                                    shuffle = False)

            y_estim = t.zeros(op_dataset.__len__(),)
            y_ground = t.zeros(op_dataset.__len__(),)
            start_idx = 0
            end_idx = hp.batch_size

            y_prev = t.zeros((1, hp.compute_dim), device = device) #(1, 128)
            for x, p, a, g, y in op_dataloader:
                with t.no_grad():
                    y_hat = model(x, p, a, g, y_prev.clone().detach())
                    
                    error_step = 0
                    for s in range(hp.num_stages):
                        error_step += loss_func(y_hat[:,:,s].squeeze(dim = -1), 
                                                y.squeeze(dim = -1))
                valid_loss[e] += error_step


                y_prev[0, -min(hp.compute_dim, x.shape[0]):] = t.argmax(y_hat[:,:,-1], 
                                            dim = 1)[-min(hp.compute_dim, x.shape[0]):]

                y_estim[start_idx:end_idx] = t.argmax(y_hat[:,:,-1].clone().detach(), dim = 1)
                y_ground[start_idx:end_idx] = y.clone().detach().squeeze()

                start_idx = end_idx
                end_idx = min(end_idx + hp.batch_size, op_dataset.__len__())

            # scores
            y_ground = y_ground.cpu().detach().numpy()
            y_estim = y_estim.cpu().detach().numpy()

            transition_idx = np.where(y_ground == 8)[0] # remove transition for evaluation
            y_estim = np.delete(y_estim, transition_idx)
            y_ground = np.delete(y_ground, transition_idx)

            valid_metrics[e, i_op, 0] = accuracy_score(y_ground, y_estim)
            h.print_log("\t\t{} Accuracy\t:{:.5f}".format(op, valid_metrics[e, i_op, 0]), log_file)
            valid_metrics[e, i_op, 3] = f1_score(y_ground, y_estim, average = 'weighted')
            h.print_log("\t\t{} F1\t:{:.5f}".format(op, valid_metrics[e, i_op, 3]), log_file)
            valid_metrics[e, i_op, 4] = jaccard_score(y_ground, y_estim, average = 'weighted')
            h.print_log("\t\t{} Jaccard\t:{:.5f}".format(op, valid_metrics[e, i_op, 4]), log_file)

        h.print_log("\n\t\tavg_acc\t:{:.5f}".format(np.mean(valid_metrics[e, :, 0])), log_file)
        h.print_log("\t\tavg_f1\t:{:.5f}".format(np.mean(valid_metrics[e, :, 3])), log_file)
        h.print_log("\t\tavg_jaccard\t:{:.5f}".format(np.mean(valid_metrics[e, :, 4])), log_file)
        
        valid_loss[e] /= len(validset)
        h.print_log("\n\t\tv_loss\t:{:.5f}\n".format(valid_loss[e]), log_file)

        # evaluate epoch
        if valid_loss[e] < best_loss + hp.escb_beta:
            best_loss = valid_loss[e]
            patience = 0
            t.save({'state_dict': model.state_dict()}, log_path + "M3X1.ckp")
            h.print_log("\t\tBest checkpoint saved", log_file)
        else:
            patience += 1
            h.print_log("\n\tPatience\t: {} !!".format(patience), log_file)

        if patience == hp.patience_lim:
            early_callback = True


h.plot_loss(train_loss.cpu().detach().numpy(), 
            valid_loss.cpu().detach().numpy(),
            log_path)

## Test
h.print_log("\n\n\ttesting...\n\n", log_file)

test_metrics = np.zeros((len(testset), 5)) # [acc, recall, precision, f1, jaccard]

for i_op, op in enumerate(testset):
    model.eval()

    op_dataset = OPDataset(op)
    op_dataloader = DataLoader(dataset = op_dataset, 
                            batch_size = hp.batch_size,
                            shuffle = False)

    y_estim = t.zeros(op_dataset.__len__(),)
    y_ground = t.zeros(op_dataset.__len__(),)
    start_idx = 0
    end_idx = hp.batch_size

    y_prev = t.zeros((1, hp.compute_dim), device = device) #(1, 128)
    for x, p, a, g, y in op_dataloader:
        with t.no_grad():
            y_hat = model(x, p, a, g, y_prev.clone().detach())

        y_prev[0, -min(hp.compute_dim, x.shape[0]):] = t.argmax(y_hat[:,:,-1], 
                                    dim = 1)[-min(hp.compute_dim, x.shape[0]):]

        y_estim[start_idx:end_idx] = t.argmax(y_hat[:,:,-1].clone().detach(), dim = 1)
        y_ground[start_idx:end_idx] = y.clone().detach().squeeze()

        start_idx = end_idx
        end_idx = min(end_idx + hp.batch_size, op_dataset.__len__())
    
    # scores
    y_ground = y_ground.cpu().detach().numpy()
    y_estim = y_estim.cpu().detach().numpy()
    
    transition_idx = np.where(y_ground == 8)[0] # remove transition for evaluation
    y_estim = np.delete(y_estim, transition_idx)
    y_ground = np.delete(y_ground, transition_idx)
    
    test_metrics[i_op, 0] = accuracy_score(y_ground, y_estim)
    h.print_log("\t\t{} Accuracy\t:{:.5f}".format(op, test_metrics[i_op, 0]), log_file)
    test_metrics[i_op, 1] = recall_score(y_ground, y_estim, average = "weighted")
    h.print_log("\t\t{} Recall\t\t:{:.5f}".format(op, test_metrics[i_op, 1]), log_file)
    test_metrics[i_op, 2] = precision_score(y_ground, y_estim, average = "weighted")
    h.print_log("\t\t{} Precision\t:{:.5f}".format(op, test_metrics[i_op, 2]), log_file)
    test_metrics[i_op, 3] = f1_score(y_ground, y_estim, average = 'weighted')
    h.print_log("\t\t{} F1\t:{:.5f}".format(op, test_metrics[i_op, 3]), log_file)
    test_metrics[i_op, 4] = jaccard_score(y_ground, y_estim, average = 'weighted')
    h.print_log("\t\t{} Jaccard\t:{:.5f}".format(op, test_metrics[i_op, 4]), log_file)

    h.plot_ribbon(np.expand_dims(y_estim, 1), op, log_path + "/results/")
    h.plot_confusion(y_estim, y_ground, op, log_path + "/results/")

h.print_log("\n\t\tAccuracy\t:{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 0]),
                                                              np.std(test_metrics[:, 0])), 
                                                              log_file)
h.print_log("\n\t\tRecall\t\t:{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 1]),
                                                              np.std(test_metrics[:, 1])), 
                                                              log_file)
h.print_log("\n\t\tPrecision\t:{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 2]),
                                                              np.std(test_metrics[:, 2])), 
                                                              log_file)
h.print_log("\n\t\tF1\t\t\t:{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 3]),
                                                              np.std(test_metrics[:, 3])), 
                                                              log_file)
h.print_log("\n\t\tJaccard\t\t:{:.5f} \u00B1 {:.5f}".format(np.mean(test_metrics[:, 4]),
                                                              np.std(test_metrics[:, 4])), 
                                                              log_file)


## Log finish time
h.print_log('\n\tTraining finished at: {} \n'.format(
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)