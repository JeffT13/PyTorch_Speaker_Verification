# SpeakerVerificationEmbedding

Implementation of an LSTM speech encoding network and [Generalized End-to-End Loss](https://arxiv.org/pdf/1710.10467.pdf). Original implementation by HarryVolek trains the `SpeechEmbedder` on the [TIMIT dataset](https://github.com/philipperemy/timit) and outlines inferring [d-vectors](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf) with it. We add the [ICSI](http://groups.inf.ed.ac.uk/ami/icsi/license.shtml) and [LibriSpeech](http://www.openslr.org/12/) datasets to the training and embed the Supreme Court of the United States (SCOTUS) [oral arguments & transcriptions](https://www.oyez.org/) into d-vectors with labels for various speaker diarization methods.

Adapted by Jeffrey Tumminia, Sophia Tsilerides, Amanda Kuznecov, Ilana Weinstein as part of NYU Center for Data Science Capstone Project. Research mentored by Prof. Aaron Kaufman. Computational resources provided by NYU Prince HPC.  


# Dependencies

* PyTorch 0.4.1
* python 3.5+
* numpy 1.15.4
* librosa 0.6.1
* webrtcvad 2.0.10


# Outline

## Datasets

We trian the `SpeechEmbedder` on following datasets

    - [TIMIT](https://github.com/philipperemy/timit)
    - [ICSI](http://groups.inf.ed.ac.uk/ami/icsi/license.shtml)
    - [LibriSpeech](http://www.openslr.org/12/)
    - [SCOTUS Oyez](https://www.oyez.org/)

Each dataset (aside from the TIMIT) contains a folder with its own readme outlining the procedures taken with it. The TIMIT dataset follows the structure of the [base repo](https://github.com/HarryVolek/PyTorch_Speaker_Verification).
    

    
## SpeechEmbedder Performance 

We evaluate our model on the original TIMIT test set after each additional dataset. Our results are below.

```
LaTeX Chart
```


## d-vector Performance 

We utilize the Oyez SCOTUS dataset to 

Inference scripts are present for the SCOTUS and Libre dataset. The preprocessing for these datasets is outlined


Calling `dvector_SCOTUS` outputs a folder for each case processed, each containing a numpy array of dvector embeddings which are unaligned with the sequence of the case audio. Calling `align_SCOTUS.py` will process these unaligned cases into a numpy array of dvectors (`case_sequence.npy`) and a numpy array of labels (`case_cluster_id.npy`) which are both the same length, as well as a csv of a list of files which were not embedded (usually do to being too short). These are formatted for [our fork of the uisrnn](https://github.com/JeffT13/LegalUISRNN)