#!/usr/bin/env python3

import glob
import csv
import os
import numpy as np
import sys
sys.path.append("./SpeakerVerificationEmbedding/src")
from hparam import hparam_Libri as hp

 
book_path = glob.glob(os.path.dirname(hp.data.save_path+'*/*'))
verbose = hp.data.verbose
  
# Reconstructs alignments and labels in order
for i, folder in enumerate(book_path):
    book = folder.split('/')[-1]
    #Skip book if already aligned
    if os.path.exists(folder+'/'+book+'_sequence.npy'):
        if verbose:
            print("Skipped book:", book)
        continue
        
    if verbose:
        print("Aligning book ", book)
      
    with open(folder+'/'+book+'_info.csv') as f:
        reader = csv.reader(f)
        path = list(reader)
    
    srtlst = sorted(path, key=lambda x: x[0])
    temp_sequence = np.load(folder+'/'+book+'_embarr.npy', allow_pickle=True)
    temp_cluster_id = np.load(folder+'/'+book+'_labelarr.npy', allow_pickle=True)
    
    sizetemp=0
    temp_lst = []
    temp_id_lst = []
    for t0,t1,s,i,j in srtlst:
        sizetemp+=int(s)
        temp_lst.append(temp_sequence[int(j)][int(i)])
        temp_id_lst.append(temp_cluster_id[int(j)][int(i)])
      
    book_emb = np.concatenate(temp_lst, axis=0)
    book_label = np.concatenate(temp_id_lst, axis=0)
      
    if verbose:
        print("Expected Sequence Shape:", sizetemp, " X ", hp.model.proj)
        print("Expected ID Shape:", sizetemp, " X ", '')
        print("Sequence Shape:", np.shape(book_emb))
        print("ID Shape:", np.shape(book_label))
        
    np.save(folder+'/'+book+'_sequence.npy', book_emb)
    np.save(folder+'/'+book+'_cluster_id.npy', book_label)
  
  
  
  
  