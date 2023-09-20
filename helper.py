from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch as t
import random
import time
import sys
import os

def print_log(text, file_name = 'log.txt', ends_with = '\n', display = False):
    '''
    Prints output to the log file.
    
    text        : string or List               
                        Output text

    file_name   : string
                        Target log file

    ends_with   : string
                        Ending condition for print func.

    display     : Bool
                        Wheter print to screen or not.
    '''
    
    if display:
        print(text, end = ends_with)

    with open(file_name, "a") as text_file:
        print(text, end = ends_with, file = text_file)


def plot_ribbon(data, title, out_path, repeat = 1024):
    ''' Plots color ribbon with legend
    
    data        : np.array [1xN]
                    Data to plot

    title       : str
                    Title and save name of the figure, e.g. OP name

    out_path    : str
                    path to save.

    repeat      : int
                    Vertical width of the ribbon 
    '''

    phases = ['Preperation', 'Puncture', 'GuideWire', 'CathPlacement', 
        'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing', 'Transition']

    # ensure data type
    assert type(data) == type(np.zeros([1, 1])), "Input data should be a numpy array"

    # ensure horizontal
    if data.shape[1] == 1:
        data = np.transpose(data)
    
    data = np.repeat(data, repeats = repeat, axis = 0)

    formatter = matplotlib.ticker.FuncFormatter(lambda s, 
        x: time.strftime('%M:%S', time.gmtime(s // 60)))
    xtick_pos = np.linspace(0, data.shape[1], data.shape[1] // 350)

    matplotlib.rc('font',family='Times New Roman')

    # discrete cmap
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 9))
    disc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('DMap', color_list, 9)

    # plot
    plt.figure(dpi = 600, figsize = (28,12))
    plt.matshow(data, cmap = disc_cmap, vmin = 0, vmax = 8)
    plt.grid(False)
    plt.yticks([])
    plt.clim(-0.5, 8.5)
    cbar = plt.colorbar(ticks = range(len(phases)))
    cbar.ax.set_yticks(np.arange(len(phases)), labels = phases, fontsize = 18)
    plt.gca().xaxis.tick_bottom()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(xtick_pos, fontsize = 18)
    plt.xlabel('Time (HH:MM)', fontsize = 18)
    plt.title(title, fontsize = 24, pad = 10)
    plt.savefig(out_path + title + ".png", bbox_inches = 'tight')
    plt.close('all')


def plot_confusion(y_estim, y_ground, title, out_path):
    conf_mat = confusion_matrix(y_ground, y_estim)
    plt.figure(dpi = 600)
    disp = ConfusionMatrixDisplay(conf_mat)
    disp.plot()
    plt.title(title)
    plt.savefig(out_path + title + "_CM.png", bbox_inches = 'tight')
    plt.close('all')


def plot_loss(train_loss, valid_loss, save_path):

    train_loss = train_loss[train_loss != 0] # cut tailing zeros after early stopper callback
    valid_loss = valid_loss[valid_loss != 0]

    plt.figure(dpi = 600, constrained_layout = True) 
    plt.style.use('fivethirtyeight')
    plt.plot(train_loss,  linewidth = 2, label = 'Train Loss')
    plt.plot(valid_loss,  linewidth = 2, label = 'Valid Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Loss Functions')
    plt.xlabel('Epochs')
    plt.savefig(save_path + '/loss.png')
    plt.close('all')


def split_dataset(dataset_path, log_file, results_file,
                  bad_ops = [], random_seed = -1, test_split = 0.2):
    '''
    Stplits dataset into train/valid/test sets
    with op-wise stratified sampling
    
    dataset_path    : string 
                        dataset path

    log_file        : string
                        path to save log file
    
    results_file    : string
                        path to save figure                    

    bad_ops         : list of strings
                        ops to exclude in partition

    random_seed     : int
                        random seed. If zero, dataset is not shuffled

    test_split      : float
                        portion of the test set, 
                        also valid set in train set
    '''
    ## Read dataset
    ops = sorted(os.listdir(dataset_path))
    if len(bad_ops) != 0:
        ops = [op for op in ops if op not in bad_ops]
    if random_seed != -1:
        random.seed(random_seed)
        random.shuffle(ops)


    ## Init variables
    trainset = dict()
    validset = dict()
    testset = dict()

    train_phase_bins = np.zeros((8, ))
    valid_phase_bins = np.zeros((8, ))
    test_phase_bins = np.zeros((8, ))

    trainset_size = int(len(ops) * (1 - 2 * test_split))
    validset_size = int(len(ops) * test_split) + 1
    testset_size = len(ops) - trainset_size - validset_size


    ## Count durations of phases in each op
    ops_dict = dict()

    total_phase_bins = np.zeros((8, )) # reference
    for i, op in enumerate(ops):
        y = t.load(dataset_path + op + "/labels")
        phase_count = t.bincount(y.int().squeeze())[:-1] # exclude transition phase
        phase_count = phase_count.squeeze().clone().cpu().numpy()
        ops_dict[op] = phase_count
        
        total_phase_bins += phase_count
        

    ## Distribute ops
    H_ops = np.zeros((len(ops_dict.keys()), ))
    for i, op in enumerate(ops_dict):
        H_ops[i] = entropy(ops_dict[op], base = 2)

    for i, (h, op) in enumerate(sorted(zip(H_ops, ops_dict.keys()), reverse = True)):
        if i % 3 == 0 and len(testset.keys()) < testset_size:
            testset[op] = ops_dict[op]
            test_phase_bins += ops_dict[op]
        elif i % 3 == 1 and len(validset.keys()) < validset_size:
            validset[op] = ops_dict[op]
            valid_phase_bins += ops_dict[op]
        else:
            trainset[op] = ops_dict[op]
            train_phase_bins += ops_dict[op]
    
    assert np.array_equal(total_phase_bins, train_phase_bins + valid_phase_bins + test_phase_bins)
    

    print_log("\tTrainset [{}] OPs\t: {}".format(len(trainset.keys()),trainset.keys()), log_file)
    print_log("\tValidset [{}] OPs\t: {}".format(len(validset.keys()), validset.keys()), log_file)
    print_log("\tTestset [{}] OPs\t\t: {}".format(len(testset.keys()), testset.keys()), log_file)

    print_log("\tTrainset Entropy\t: {}".format(entropy(train_phase_bins)), log_file)
    print_log("\tValidset Entropy\t: {}".format(entropy(valid_phase_bins)), log_file)
    print_log("\tTestset Entropy\t: {}".format(entropy(test_phase_bins)), log_file)


    phases = ['Preperation', 'Puncture', 'GuideWire', 'CathPlacement', 
        'CathPositioning', 'CathAdjustment', 'CathControl', 'Closing']

    matplotlib.rc('font',family='Times New Roman')
    
    def_cmap = plt.cm.get_cmap('tab10')
    color_list = def_cmap(np.linspace(0, 1, 9))

    fig, axs = plt.subplots(3, 1, sharex = True, figsize = (16, 12), dpi = 600)

    axs[0].bar(range(8), train_phase_bins / total_phase_bins, color = color_list)
    axs[0].set_title("Training Set [{} OPs]".format(trainset_size), fontsize = 26)
    axs[0].set_yticks(np.linspace(0, 1, num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[0].margins(x = 0.01)

    axs[1].bar(range(8), valid_phase_bins / total_phase_bins, color = color_list)
    axs[1].set_title("Validation Set [{} OPs]".format(validset_size), fontsize = 26)
    axs[1].set_yticks(np.linspace(0, 1, num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[1].margins(x = 0.01)

    axs[2].set_title("Test Set [{} OPs]".format(testset_size), fontsize = 26)
    axs[2].bar(range(8), test_phase_bins / total_phase_bins, color = color_list)
    axs[2].set_yticks(np.linspace(0, 1, num = 5), labels = np.linspace(0, 1, num = 5) ,fontsize = 20)
    axs[2].set_xticks(np.arange(len(phases)), labels = phases, fontsize = 20, rotation = 15)
    axs[2].margins(x = 0.01)

    plt.savefig(results_file + "/class_dist.png", bbox_inches = 'tight')
    plt.close('all')

    return list(trainset.keys()), list(validset.keys()), list(testset.keys())