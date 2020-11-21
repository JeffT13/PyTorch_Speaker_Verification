# SCOTUS_Speaker_Verification


Implementation of speech embedding net and loss (described [here](https://arxiv.org/pdf/1710.10467.pdf)). Original implementation by HarryVolek utilizes the [TIMIT dataset](https://github.com/philipperemy/timit) for training the speech embedder. We add the [ISCI database]() to the speech embedding net training and convert the Supreme Court of the United States (SCOTUS) oral arguments into a d-vectors for the speaker diarization model found at [uisrnn](https://github.com/google/uis-rnn). Dataset parameters are handled in isolated config files.   

Adapted by Sophia Tsilerides, Jeffrey Tumminia, Amanda Kuznecov, Ilana Weinstein as part of NYU Center for Data Science Capstone Project. Research mentored by Prof. Aaron Kaufman. Computational resources provided by NYU Prince HPC.  


# Dependencies

* PyTorch 0.4.1
* python 3.5+
* numpy 1.15.4
* librosa 0.6.1
* webrtcvad 2.0.10 (necc for dvectors)


# `SpeechEmbedder` Training

We follow the original implementation by HarryVolek to train the speech embedding network on the TIMIT dataset. Instructions for data preprocessing and training can be found in TIMIT README (which is the original repo README).   

### ISCI

follows similar implementation to TIMIT. Uses l `config/...`

## Performance 

```
EER across # epochs:
```


# SCOTUS processing


### data processing


### dvector embedding creations




This procedure outputs to a folder called `SCOTUS_Processed` a folder for each case processed, each containing a numpy array of dvector embeddings (`case_sequence.npy`) and a numpy array of labels (`case_cluster_id.npy`) which are both the same length, as well as a csv of a list of files which were not embedded (usually do to being too short). These are formatted for [our fork of the uisrnn](https://github.com/JeffT13/uis-rnn) 
