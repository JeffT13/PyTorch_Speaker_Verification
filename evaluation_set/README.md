## Evaluation Set

### Details
Using LibriSpeech3 dataset for evaluating diarization model (https://github.com/EMRAI/emrai-synthetic-diarization-corpus) . 
3-person dialogues without overlap were used in the evaluation since it best represented the SCOTUS dialogues (a few speakers throughout with no overlap). Audio and diarization was used from the subfolder: https://github.com/EMRAI/emrai-synthetic-diarization-corpus/tree/master/librispeech3/test_clean

### Prerequisites 

####Downloading LibriSpeech3

Most of our computing is done on an HPC that does not support the requests package and therefore downloading the LibriSpeech3 data was done locally, and files are transferred to the HPC cluster manually.

The specific folder containing the test set was downloaded using sparse checkout. This can be done using the following steps dervied from https://stackoverflow.com/questions/33066582/how-to-download-a-folder-from-github?noredirect=1&lq=1:

1.  Create a directory
    mkdir librispeech
    cd librispeech
    
2.  Set up a git repo
    git init
    git remote add origin https://github.com/EMRAI/emrai-synthetic-diarization-corpus

3.  Configure git-repo to download only specific directories
    git config core.sparseCheckout true

4.  Set the folder to the test_clean folder (do this step for both audio files and ctms (diarizations)
    echo "librispeech3/test_clean/wavs" > .git/info/sparse-checkout 
    echo "librispeech3/test_clean/ctms" > .git/info/sparse-checkout

5.  Download repo
    git pull origin master
    
####HPC Prerequisities

All the modules in HPC needed for this process are:

`module purge`

`module load python3/intel/3.7.3` 

`conda install -c auto pydub`

`module swap anaconda3  python/intel/2.7.12`

`module load librosa/intel/0.5.0`

### Pre-Processing Instructions

1.  Place downloded LibriSpeech3 audio .wav files in `./librispeech/audio` folder on HPC. 

2.  Place downloaded LibriSpeech3 diarized .ctm files in `./librispeech/diarization` folder on HPC.

3.  Place `librispeech_process.py` script in `./librispeech` folder.

4.  Run `./librispeech_process.py' to create `speakers` folder to store each speaker as a subfolder containing [start_time end_time] as .txt file, and corresponding audio files for audio segments as .wav.
