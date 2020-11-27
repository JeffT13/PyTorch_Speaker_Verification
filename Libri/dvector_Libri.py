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

from hparam import hparam_Libri as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
from utils import concat_segs, get_STFTs, align_embeddings

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
    
    #Skip book if already processed
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
            if False: #for debugging
                print('processing file', file)
                print('fullpath', folder+'/'+spkr_name+'/'+file)
            if file[-4:] == '.wav':
                times, segs = VAD_chunk(2, folder+'/'+spkr_name+'/'+file)
                # Bad .wav detection
                if segs == []:
                    if verbose:
                        print("Bad wav")
                    rm_pthlst.append(folder+'/'+file)
                    continue

                concat_seg = concat_segs(times, segs)
                if len(concat_seg)<min_va:
                    if verbose:
                        print("short wav")
                    rm_pthlst.append(folder+'/'+file)
                    continue

                STFT_frames = get_STFTs(concat_seg)
                STFT_frames = np.stack(STFT_frames, axis=2)
                STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
                STFT_frames = STFT_frames.to(hp.device)
                embeddings = embedder_net(STFT_frames)
                aligned_embeddings = align_embeddings(embeddings.detach().cpu().numpy())
                if verbose:
                    print('shape:', np.shape(aligned_embeddings))

                spkr_sequence.append(aligned_embeddings)
                spkr_cluster_id = []
                for embedding in aligned_embeddings:
                    spkr_cluster_id.append(str(use_label)) #use_label handling judge id

                #Track full names of processed wav files
                pth = file.split(".")[0]+'.txt'
                pth = bk+'/'+spkr_name+'/'+pth
                f = open(hp.data.main_path+pth, 'r')
                f = f.read().split(" ")
                spkr_file_lst.append((f[0], f[1], np.shape(aligned_embeddings)[0], filecount, spkrtracker))
                spkr_cluster_lst.append(spkr_cluster_id)
                filecount = filecount + 1          

        if verbose:
            print('Processed', filecount, 'files for Book', bk, 'for spkr', spkr_name)
        bk_file_lst.append(spkr_file_lst)
        bk_sequence.append(spkr_sequence)
        bk_cluster_id.append(spkr_cluster_lst)
        spkrtracker+=1

    if verbose:
        print('Handled', spkrtracker, 'speakers for book', bk)
        print('saving bk sequence', bk)

    fold = hp.data.save_path+bk+'/'
    if not os.path.exists(fold):
        os.makedirs(fold)
    temp_sequence = np.asarray(bk_sequence, dtype='object')
    temp_cluster_id = np.asarray(bk_cluster_id, dtype='object')
    np.save(fold+bk+'_embarr',temp_sequence)
    np.save(fold+bk+'_labelarr',temp_cluster_id)


    info_lst=[item for sublist in bk_file_lst for item in sublist]
    with open(fold+bk+'_info.csv', 'w+') as file:     
        write = csv.writer(file) 
        write.writerows(info_lst)

    with open(fold+bk+'_2remove.csv', 'w') as rm:
        wr = csv.writer(rm, delimiter="\n")
        wr.writerow(rm_pthlst)
