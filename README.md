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

We train the `SpeechEmbedder` on following datasets

    - [TIMIT](https://github.com/philipperemy/timit)
    - [ICSI](http://groups.inf.ed.ac.uk/ami/icsi/license.shtml)
    - [LibriSpeech](http://www.openslr.org/12/)

Each dataset (aside from the TIMIT) contains a folder with its own readme outlining the procedures taken with it. The TIMIT dataset follows the structure of the [base repo](https://github.com/HarryVolek/PyTorch_Speaker_Verification). The README of the SpeechEmbedder folder is a slight variation of the README from the base repo including our test performance. 
    
We then evaluate the performance of the embedding network for d-vector generation with the [SCOTUS Oyez](https://www.oyez.org/) dataset which has been [https://github.com/walkerdb/supreme_court_transcripts](diarized and transcribed). The SCOTUS folder contains a README outlining the contents of the folder and the audio processing + d-vector generation procedure.

    