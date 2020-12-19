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

#-----------------------------
# Following functions adapted/introduced in fork
def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    hold_times = []
    t0 = 0
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            hold_times.append((t0, times[i][1]))
            t0 = times[i+1][0]
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
        hold_times.append((t0, times[-1][1]))
    return concat_seg, hold_times

def get_STFTs(segs,htemp):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    STFT_labels = []
    idx = 0
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
                STFT_labels.append(htemp[idx])
            else:
                break
        idx+=1
    return STFT_frames, STFT_labels
    
def align_embeddings(embeddings, labs):
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
    emb_labels = []
    for i, partition in enumerate(partitions):
        emb_lab = labs[partition[0]:partition[1]]
        if  len(set(emb_lab))>1:
          continue
        else:
          avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
          emb_labels.append(emb_lab[0])        
    return avg_embeddings[0:len(emb_labels)], emb_labels

#new function
def align_times(casetimelist, hold_times, spkr_dict):
  htemp = []
  _, endtime, endspkr = casetimelist[-1]
  for i, h in enumerate(hold_times):
    append = False
    if h[1]>=endtime:
        htemp.append(spkr_dict[endspkr])
        append = True
    else:
      for j, c in enumerate(casetimelist):
        if h[1]<c[1]:
          if h[1]>=c[0]:
            spkr_name = c[2]
            htemp.append(spkr_dict[spkr_name])
            append = True
          elif h[1]<c[0] and not append:
            htemp.append(spkr_dict[casetimelist[j-1][2]])
            append = True
          elif not append:
            print("labelling overlapping at ", h)
            htemp.append(999) #overlap in diarization and VAD
            apend=True
    if not append and h[1]!=hold_times[-1][1]:
      print('value not appended in loop')
      print(i, h)
  return htemp
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
      STFT_samp = STFT_samp.to(hp.device)
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


    
