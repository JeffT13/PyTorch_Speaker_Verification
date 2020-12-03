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


#Build judge dictionary
import glob
import numpy as np
import os
import json
import sys
sys.path.append("./SpeakerVerificationEmbedding/src")
from hparam import hparam_SCOTUS as hp

case_path = glob.glob(os.path.dirname(hp.unprocessed_data))
label = 20 # unknown speaker label counter (leave room for 20 judges)
cnt = 0 # counter for judge_dict
spkr_dict = dict()
casetimedict = dict()

if os.path.exists(hp.data.dict_path+'casetimes.json'):
    print("Dictionaries already saved down")
else:
    # Build case info dictionary
    for i, path in enumerate(case_path):
      file = path.split('/')[-1]
      if file[-4:] == '.txt':
        filetimelist= []
        f= open(path,'r')
        k=f.readlines()
        f.close()
        for u in k:
          t0, t1, spkr = u.split(' ')[0:3]
          #build speaker dictionary
          if spkr[-14:]=='scotus_justice':
            if spkr not in spkr_dict:
              spkr_dict[spkr] = cnt
              cnt+=1
              if cnt>=20:
                print("ERROR: NEED MORE JUDGE ROOM")
          else:
            if spkr not in spkr_dict:
              spkr_dict[spkr] = label
              label+=1         
          filetimelist.append((float(t0),float(t1),spkr))
        casetimedict[file[:-4]] = filetimelist

    #save spkr label dictionary
    #PATH MANUAL (will adapt in full run)
    with open(hp.data.dict_path+'spkrs.json', 'w') as outfile:  
        json.dump(spkr_dict, outfile) 
    
    with open(hp.data.dict_path+'casetimes.json', 'w') as outfile:  
        json.dump(casetimedict, outfile)

