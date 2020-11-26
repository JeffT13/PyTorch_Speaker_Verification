#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 16:56:19 2018

@author: harry
"""

import sys
sys.path.append("./SpeakerVerificationEmbedding/src")
print(sys.path)

import glob
import librosa
import numpy as np
import os
import torch
import json
import csv

print("starting custom imports")

from hparam import hparam_SCOTUS as hp
from VAD_segments import VAD_chunk
from utils import get_centroids, get_cossim, calc_loss
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from utils import concat_segs, get_STFTs, align_embeddings
print("imports complete")

print("initialzie speech embedder")
embedder_net = SpeechEmbedder()
print(hp.model.model_path)
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.to(hp.device)
print("completed embedder init")