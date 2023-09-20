""" 
    This scripts extracts features and saves each OP
    into a file"""

import torchaudio.transforms as T
import torch.nn.functional as F
import torchxrayvision as xrv
import pandas as pd
import numpy as np
import torchaudio
import torch
import json
import sys
import os

from skimage.io import imread


# functions
mfcc_transform = T.MFCC(sample_rate = 16000, n_mfcc = 34, 
                        melkwargs = {"n_fft":2048,
                                     "n_mels":128,
                                     "hop_length":160, # 10ms
                                     "win_length":640  # 40ms
                                     })

spectrogram_transform = T.Spectrogram(n_fft=2048,
                            win_length=640,
                            hop_length=160,
                            center=True,
                            pad_mode="reflect",
                            power=2.0)

def time2sec(time_str):
    time_str = time_str.split(':')
    if len(time_str) == 3:
        sec = int(time_str[0]) * 3600 + \
                int(time_str[1]) * 60 + int(time_str[2])
    elif len(time_str) == 2:
        sec = int(time_str[0]) * 60 + int(time_str[1])
    return sec

def final_second(x):
    if not isinstance(x.shape[1] / 16000, int):
        target_len = int(np.ceil(x.shape[1] / 16000) * 16000)
        return F.pad(x, (0, target_len - x.shape[1]), "constant", 0)
    else:
        return x

def mffc_feats(x):
    # compute mfccs
    mfccs = mfcc_transform(x).squeeze().T

    # chunk
    num_win = int((mfccs.shape[0] - 1) / 100)
    mfcc_chunks = torch.zeros(num_win, 100, mfccs.shape[1])
    start_idx = 0
    end_idx = 100
    for i in range(num_win):
        mfcc_chunks[i, :, :] = mfccs[start_idx:end_idx, :]
        start_idx = end_idx
        end_idx += 100
    
    return mfcc_chunks


def spec_feats(x):
    # compute spectrograms
    S = spectrogram_transform(x)

    # chunk
    num_win = int((S.shape[2] - 1) / 100)
    spec_chunks = torch.zeros(num_win, 1024, 100)
    start_idx = 0
    end_idx = 100
    for i in range(num_win):
        spec_chunks[i, :, :] = S[0, :-1, start_idx:end_idx]
        start_idx = end_idx
        end_idx += 100
    
    return spec_chunks

def wav2vec_feats(x, wav2vec_model, win_len = 3, hop_len = 1, 
                  sampling_rate = 16000):
    # non-casual [--- x ---]
    # fps = 1 (hop_length?)

    # init
    num_win = int(x.shape[1] / sampling_rate)
    features = torch.zeros(num_win, win_len * 50 - 1, 1024)

    # calculate variables
    pad_width = int((win_len - 1) / 2 * sampling_rate)
    win_len *= sampling_rate
    hop_len *= sampling_rate
    x = F.pad(x, (pad_width, pad_width), "constant", 0)

    # iter
    start_idx = 0
    end_idx = win_len
    for i in range(num_win):
        x_chunk = x[:, start_idx:end_idx].clone().detach()
        x_feat, _ = wav2vec_model.extract_features(x_chunk)
        features[i, :, :] = x_feat[-1].clone().detach()
        start_idx += hop_len 
        end_idx += hop_len

    return features

# paths
audio_path = "/DATA/dataset_belen/Audio/"
frames_path = "/DATA/dataset_belen/Frames/"
annotation_file = "/DATA/dataset_belen/annotations.json"
target_path = "/DATA/dataset_belen/Dataset_Wav2vec20_3s/"
if not os.path.exists(target_path):
    os.mkdir(target_path)

# read annotations
with open(annotation_file) as f:
    annots = json.load(f)

# ops 
ops = sorted(os.listdir(frames_path))
ops.remove('OP_018')

# Label dictionary
label_dic = {'Preparation' : 0, 'Puncture' : 1, 'GuideWire' : 2, 
    'CathPlacement' : 3, 'CathPositioning' : 4, 'CathAdjustment' : 5, 
    'CathControl' : 6, 'Closing' : 7, 'Transition' : 8}

# feature extraction models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
wav2vec = bundle.get_model().to(device)
xrayvision = xrv.models.DenseNet(weights="densenet121-res224-all")


# iter
for op in ops:
    print("Working on...", op)
    

    # find surgeon/assistant channels
    for annot in annots:
         if annot[0] == op:
            ch1_owner = annot[1]
            ch2_owner = annot[2]
            print("\tFirst audio channel\t:", ch1_owner)               
            print("\tSecond audio channel\t:", ch2_owner)


    # get frames
    frames = sorted([frames_path + op + "/" + f for f in os.listdir(frames_path + op)])
    print("\tOP has {} frames".format(len(frames)))


    # read annotation file
    labels = np.ones((len(frames), 1)) * 99

    for annot in annots:
        if annot[0] == op:
            time_stamps = annot[3]
            phases = annot[4]
            for p, t in zip(phases, time_stamps):
                t = t.split('-')
                t_start = time2sec(t[0])
                t_end = time2sec(t[1]) + 1
                if p == 'Closing':
                    labels[t_start:] = label_dic[p]
                else: 
                    labels[t_start:t_end] = label_dic[p]
    labels = torch.from_numpy(labels)
    

    # compute x-ray features
    xray_features = torch.zeros(len(frames), 1024)
    for i,f in enumerate(frames):
        print("\tProcess {:.2f} %".format(i / len(frames) * 100), end="\r")
        im = imread(f)
        im = xrv.datasets.normalize(im, 255)
        im = im.mean(2)[None, ...]
        im = torch.from_numpy(im)
        out = xrayvision.features2(im[None,...])
        xray_features[i, :] = out.detach()
    print("\tX-Ray Features are extracted!")
    

    # compute audio features
    if ch1_owner == "physician":
        physician_mic, _ = torchaudio.load(audio_path + op + "_Channel_1.wav")
        physician_mic = final_second(physician_mic)
        assistant_mic, _ = torchaudio.load(audio_path + op + "_Channel_2.wav")
        assistant_mic = final_second(assistant_mic)

    elif ch1_owner == "assistant":
        physician_mic, _ = torchaudio.load(audio_path + op + "_Channel_2.wav")
        physician_mic = final_second(physician_mic)  
        assistant_mic, _ = torchaudio.load(audio_path + op + "_Channel_1.wav")
        assistant_mic = final_second(assistant_mic)
    
    ambient_mic, _ = torchaudio.load(audio_path + op + "_Channel_3.wav")
    ambient_mic = final_second(ambient_mic)

    S_p = wav2vec_feats(physician_mic.to(device), wav2vec, 3, 1)
    S_a = wav2vec_feats(assistant_mic.to(device), wav2vec, 3, 1)
    S_g = wav2vec_feats(ambient_mic.to(device), wav2vec, 3, 1)
    print("\tAudio Features are extracted!")


    # save to a file
    target_file = target_path + op + '/'
    os.mkdir(target_file)

    torch.save(xray_features, target_file + "x_ray")
    torch.save(S_p, target_file + "physician_mic")
    torch.save(S_p, target_file + "assistant_mic")
    torch.save(S_a, target_file + "ambient_mic")
    torch.save(S_g, target_file + "physician_mic")
    torch.save(labels, target_file + "labels")
    
    print("\tOP data saved to\t:", target_file)