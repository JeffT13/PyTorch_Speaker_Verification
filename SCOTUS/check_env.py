#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import sys
#sys.path.append("./SpeakerVerificationEmbedding/src")
#print(sys.path)

import glob
import librosa
import webrtcvad
import wave
import numpy as np
import os
import torch
import json
import csv

print("starting custom imports")

from SpeechEmbedder.hparam import hparam as hp_test
from SpeechEmbedder.VAD_segments import VAD_chunk
from SpeechEmbedder.utils import get_centroids, get_cossim, calc_loss
from SpeechEmbedder.speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from SpeechEmbedder.utils import concat_segs, get_STFTs, align_embeddings
print("imports complete")

print("initialzie speech embedder")
from SpeechEmbedder.hparam import hparam_SCOTUS as hp
embedder_net = SpeechEmbedder()
print(hp.model.model_path)
embedder_net.load_state_dict(torch.load(hp.model.model_path))
embedder_net.to(hp.device)
print("completed embedder init")