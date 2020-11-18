#!/usr/bin/env python3

import globa
import json
import os
import numpy as np

from hparam import hparam_SCOTUS as hp

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