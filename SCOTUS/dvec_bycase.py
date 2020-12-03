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
sys.path.append("./SpeakerVerificationEmbedding/src")
from hparam import hparam_SCOTUS as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
from utils import concat_segs, get_STFTs, align_embeddings, align_times
 
#-----------------------------
    
#initialize SpeechEmbedder
embedder_net = SpeechEmbedder()
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.to(hp.device)
embedder_net.eval()


case_path = glob.glob(os.path.dirname(hp.unprocessed_data))

with open(hp.data.dict_path+'spkrs.json') as json_file: 
    spkr_dict = json.load(json_file)
    
with open(hp.data.dict_path+'casetimes.json') as json_file: 
    casetimedict = json.load(json_file)


fold = hp.data.save_path
cut_div = 4
train_sequences = []
train_cluster_ids = []
print("starting generation")

for i, path in enumerate(case_path):
  
  file = path.split('/')[-1]
  if os.path.exists(fold+file[:-4]+'_seq.npy'):
    print('skipped ', file[:-4])
    continue
  if file[-4:] == '.wav':
    times, segs = VAD_chunk(2, path)
    concat_seg, ht = concat_segs(times, segs)
    
    # ---
    #assumes files will precleaned
    #need to improve for any full case set processing
    htemp = align_times(casetimedict[file[:-4]], ht, spkr_dict)
    if hp.data.verbose: 
      #if len(ht)!= len(htemp) then
      # skip case and track case_id
      print(len(ht))
      print(len(htemp))
    # ---  
      
    STFT_frames, STFT_labels = get_STFTs(concat_seg, htemp)
    STFT_frames = np.stack(STFT_frames, axis=2)
    STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
 
    cut = STFT_frames.size()[0]//cut_div
    t0 = 0
    temp_emb = []
    for i in range(cut_div):
      if i<(cut_div-1):
        STFT_samp = STFT_frames[t0:t0+cut, :, :]
      else:
        STFT_samp = STFT_frames[t0:, :, :]
      #process slice
      STFT_samp = STFT_samp.to(device)
      emb = embedder_net(STFT_samp)
      temp_emb.append(emb.detach().cpu().numpy())
      t0+=cut

    embeddings = np.concatenate(temp_emb, axis=0)
    print(embeddings.shape, len(STFT_labels))
    aligned_embeddings, aligned_labels = align_embeddings(embeddings, STFT_labels)
    train_sequences.append(aligned_embeddings)
    train_cluster_ids.append(aligned_labels)
    np.save(fold+file[:-4]+'_seq',  aligned_embeddings)
    np.save(fold+file[:-4]+'_id', aligned_labels)
    print('values appended')

full_set = False
if full_set:
    np.save('/scratch/jt2565/train_seq', train_sequences)
    np.save('/scratch/jt2565/train_clus', train_cluster_ids)


    
