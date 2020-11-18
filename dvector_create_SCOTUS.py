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

"""

import glob
import librosa
import numpy as np
import os
import torch
import json
import csv

from hparam import hparam_SCOTUS as hp
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
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.to(hp.device)

#dataset path
case_path = glob.glob(os.path.dirname(hp.unprocessed_data))

total_case_num = len(case_path)
train_case_num= (total_case_num//10)*9 #90% training
min_va = 2 # minimum voice activity length
label = 20 # unknown speaker label counter (leave room for 20 judges)
train_saved = False

cnt = 0 # counter for judge_dict
judge_dict = dict()

rm_pthlst = [] #list of wav files too short to process 
train_sequence = [] # sequence holder
train_cluster_id = [] # cluster_id holder

# File Use Tracking
trn_pthlst = dict() 
tst_pthlst = dict() 

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
for i, folder in enumerate(case_path):
  case = folder.split('/')[-1]
  if verbose:
    print("Processing case:", case)

  case_file_lst = []
  case_sequence = []
  case_cluster_id = []
  s=0
  
  for spkr_name in os.listdir(folder):
    count = 0
    if verbose:
      print("Processing spkr:", spkr_name)

    #Build Judge Dictionary & Handle Other
    if spkr_name[-14:]=='scotus_justice':
      if spkr_name[:-15] in judge_dict:
        use_label = judge_dict[spkr_name[:-15]]
      else:
        judge_dict[spkr_name[:-15]] = cnt
        use_label = judge_dict[spkr_name[:-15]]
        cnt+=1
    else:
      use_label = label 
      label+=1

    spkr_file_lst = []
    spkr_sequence = []
    spkr_cluster_lst = []
    spkr_cluster_id = []

    for file in os.listdir(folder+'/'+spkr_name):
      if file[-4:] == '.wav':
        times, segs = VAD_chunk(2, folder+'/'+spkr_name+'/'+file)

        # Bad .wav detection
        if segs == []:
          #print('No voice activity detected')
          rm_pthlst.append(folder+'/'+file)
          continue

        concat_seg = concat_segs(times, segs)
        if len(concat_seg)<min_va:
          #print('Below Minimum voice activity detected')
          rm_pthlst.append(folder+'/'+file)
          continue
           
        STFT_frames = get_STFTs(concat_seg)
        STFT_frames = np.stack(STFT_frames, axis=2)
        STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0)))
        STFT_frames = STFT_frames.to(hp.device)
        embeddings = embedder_net(STFT_frames)
        aligned_embeddings = align_embeddings(embeddings.detach().cpu().numpy())
        
        spkr_sequence.append(aligned_embeddings)
        for embedding in aligned_embeddings:
          spkr_cluster_id.append(str(use_label)) #use_label handling judge id

        #Track full names of processed wav files
        pth = file.split(".")[0]+'.txt'
        pth = case+'/'+spkr_name+'/'+pth
        f = open(hp.data.main_path+pth, 'r')
        f = f.read().split(" ")
        spkr_file_lst.append((f[0], f[1], np.shape(aligned_embeddings)[0], count, s))
        spkr_cluster_lst.append(spkr_cluster_id)
        count = count + 1

    if verbose:
      print('Processed', count, 'files for case', case, 'for spkr', spkr_name)
    case_file_lst.append(spkr_file_lst)
    case_sequence.append(spkr_sequence)
    case_cluster_id.append(spkr_cluster_lst)
    s+=1

  if verbose:
    print('Handled', s, 'speakers for case', case)
  if i >= train_case_num:
    train_saved = True

  if not train_saved:
    print('saving train case sequence', case)
    trn_pthlst[case]=[item for sublist in case_file_lst for item in sublist]
    train_sequence = np.asarray(case_sequence, dtype='object')
    train_cluster_id = np.asarray(case_cluster_id, dtype='object')
    train_fold = hp.data.train_path+case+'/'
    if not os.path.exists(train_fold):
      os.makedirs(train_fold)
    np.save(train_fold+case+'_seq',train_sequence)
    np.save(train_fold+case+'_id',train_cluster_id)
    train_sequence = []
    train_cluster_id = []
  else:
    print('saving test case sequence', case)
    tst_pthlst[case]=[item for sublist in case_file_lst for item in sublist]
    train_sequence = np.asarray(case_sequence, dtype='object')
    train_cluster_id = np.asarray(case_cluster_id, dtype='object')
    test_fold = hp.data.test_path+case+'/'
    if not os.path.exists(test_fold):
      os.makedirs(test_fold)
    np.save(test_fold+case+'_seq',train_sequence)
    np.save(test_fold+case+'_id',train_cluster_id)
    train_sequence = []
    train_cluster_id = []


dictpath = hp.data.dict_path

trainjson = json.dumps(trn_pthlst)
j = open(hp.data.dict_path+"traininfo_dict.json", "w")
j.write(trainjson)
j.close()

testjson = json.dumps(tst_pthlst)
k = open(hp.data.dict_path+"testinfo_dict.json", "w")
k.write(testjson)
k.close()

with open(hp.data.dict_path+'rm_pthlst.csv', 'w') as r:
  wr = csv.writer(r, delimiter=",")
  wr.writerow(rm_pthlst)



# Could make from here below a seperate file
with open(hp.data.dict_path+"traininfo_dict.json") as jsonline:
  train_pathlist = json.load(jsonline)
with open(hp.data.dict_path+"testinfo_dict.json") as jsonline:
  test_pathlist = json.load(jsonline)
  
# Reconstructs alignments and labels in order
for dir in [hp.data.train_path, hp.data.test_path]:
  file_path = glob.glob(os.path.dirname(dir+"*/"))

  # reconstructed temp seq in order
  for i, folder in enumerate(file_path):
    siz = 0
    case = folder.split('/')[-1]
    if dir.split('/')[-2]=='train':
      path = train_pathlist[case]
    elif dir.split('/')[-2]=='test':
      path = test_pathlist[case]
    else:
      print("Bad dir")
      raise RuntimeError

    #sorts sequence by times 
    srtlst = sorted(path, key=lambda x: x[0])
    
    temp_sequence = np.load(dir+case+'/'+case+'_seq.npy',allow_pickle=True)
    temp_cluster_id = np.load(dir+case+'/'+case+'_id.npy', allow_pickle=True)
    temp_lst = []
    temp_id_lst = []

    for t0,t1,s,i,j in srtlst:
      siz+=s
      temp_lst.append(temp_sequence[j][i])
      temp_id_lst.append(temp_cluster_id[j][i])

    case_emb = np.concatenate(temp_lst, axis=0)
    print("Expected Shape:", siz, " X ", 256)
    print(np.shape(case_emb))
    case_label = np.concatenate(temp_id_lst, axis=0)

    np.save(dir+case+'/'+case[:-7]+'_emb.npy', case_emb)
    np.save(dir+case+'/'+case[:-7]+'_label.npy', case_label)
