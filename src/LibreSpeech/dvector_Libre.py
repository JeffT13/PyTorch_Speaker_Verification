#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:34:01 2018

@author: Harry

Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn


Edited on Nov 11, 2020
@author: Jeffrey Tumminia
Create dvector embeddings and labels 
based on SCOTUS Oyex data (preprocessed)

Outputs 2 json (training & testing) 
+ 1 csv (bad .wav)

"""

import glob
import librosa
import numpy as np
import os
import torch
import json
import csv
import sys

#assumes you are calling SVE repo from outside (ie LegalSpeech repo)
sys.path.append("./SpeakerVerificationEmbedding/src")
from hparam import hparam_Libre as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk


def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
            else:
                break
    return STFT_frames

def align_embeddings(embeddings):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
    return avg_embeddings
    

#initialize SpeechEmbedder
embedder_net = SpeechEmbedder()
print(hp.model.model_path)
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.to(hp.device)

#dataset path
bk_path = glob.glob(os.path.dirname(hp.unprocessed_data))

min_va = 2 # minimum voice activity length
label = 0 # unknown speaker label counter

verbose = hp.data.verbose
embedder_net.eval()


'''

SCOTUS Processing Loop 
Saves 
    Utterance Representations .npy
    Utterance Label .npy
    Time, Utt Length, location 
        - dictionaries (.json)
    bad wav file list 
        - .csv
'''

for i, folder in enumerate(bk_path):
    bk = folder.split('/')[-1]
    
    #Skip case if already processed
    if os.path.exists(hp.data.save_path+bk):
        if verbose:
            print("Skipped bk:", bk)
        continue
        
    if verbose:
        print("Processing bk:", bk)

    rm_pthlst = []
    bk_file_lst = []
    bk_sequence = []
    bk_cluster_id = []
    spkrtracker=0

    for spkr_name in os.listdir(folder):
        filecount = 0
        if verbose:
            print("Processing spkr:", spkr_name)
        use_label = label 
        label+=1

        spkr_file_lst = []
        spkr_sequence = []
        spkr_cluster_lst = []

        for file in os.listdir(folder+'/'+spkr_name):
            if file[-4:] == '.wav':
                times, segs = VAD_chunk(2, folder+'/'+spkr_name+'/'+file)

                # Bad .wav detection
                if segs == []:
                    rm_pthlst.append(folder+'/'+file)
                    continue

                concat_seg = concat_segs(times, segs)
                if len(concat_seg)<min_va:
                    rm_pthlst.append(folder+'/'+file)
                    continue

                STFT_frames = get_STFTs(concat_seg)
                STFT_frames = np.stack(STFT_frames, axis=2)
                STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
                STFT_frames = STFT_frames.to(hp.device)
                embeddings = embedder_net(STFT_frames)
                aligned_embeddings = align_embeddings(embeddings.detach().cpu().numpy())

                spkr_sequence.append(aligned_embeddings)
                spkr_cluster_id = []
                for embedding in aligned_embeddings:
                    spkr_cluster_id.append(str(use_label)) #use_label handling judge id

                #Track full names of processed wav files
                pth = file.split(".")[0]+'.txt'
                pth = case+'/'+spkr_name+'/'+pth
                f = open(hp.data.main_path+pth, 'r')
                f = f.read().split(" ")
                spkr_file_lst.append((f[0], f[1], np.shape(aligned_embeddings)[0], filecount, spkrtracker))
                spkr_cluster_lst.append(spkr_cluster_id)
                filecount = filecount + 1

        if verbose:
            print('Processed', filecount, 'files for case', bk, 'for spkr', spkr_name)
        bk_file_lst.append(spkr_file_lst)
        bk_sequence.append(spkr_sequence)
        bk_cluster_id.append(spkr_cluster_lst)
        spkrtracker+=1

    if verbose:
        print('Handled', spkrtracker, 'speakers for case', case)
        print('saving case sequence', case)

    fold = hp.data.save_path+case+'/'
    if not os.path.exists(fold):
        os.makedirs(fold)
    temp_sequence = np.asarray(case_sequence, dtype='object')
    temp_cluster_id = np.asarray(case_cluster_id, dtype='object')
    np.save(fold+case+'_embarr',temp_sequence)
    np.save(fold+case+'_labelarr',temp_cluster_id)


    info_lst=[item for sublist in case_file_lst for item in sublist]
    with open(fold+case+'_info.csv', 'w+') as file:     
        write = csv.writer(file) 
        write.writerows(info_lst)

    with open(fold+case+'_2remove.csv', 'w') as rm:
        wr = csv.writer(rm, delimiter=",")
        wr.writerow(rm_pthlst)
