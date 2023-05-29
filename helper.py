from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch as t
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