### Details
Preparing ICSI Meeting Corpus for speech embedder training/testing.

ICSI Meeting Corpus License: http://groups.inf.ed.ac.uk/ami/icsi/license.shtml

### Prerequisites 

Most of our computing is done on an HCP that does not support the requests package and therefore the script `get_icsi_data.py` is done locally, and files are transferred to the HPC cluster manually.

All the modules in HPC needed for this process are:

`module purge`

`module load python3/intel/3.7.3` 

`conda install -c auto pydub`

`conda install -c conda-forge pyyaml`

`module swap anaconda3  python/intel/2.7.12`

`module load librosa/intel/0.5.0`

### Instructions

1.  Extract raw transcript and audio files with `get_icsi_data.py` locally. Transfer `icsi_data` folder, containing subfolders `raw_transcripts` and `raw_audio`    to HPC cluster.

2.  Run `create_icsi_dataset.py` to create speaker folders for every speaker encountered, located as `./icsi_data/icsi_files`. Each speaker folder will contain [start_time end_time transcript_text] as .txt file, and corresponding audio files for audio segments as .wav.

3.  Run `del_small_files.py` to remove .wav files that are <40KB, as well as with their corresponding .txt files by running `del_small_files.py`. These audio files are too small to process; buffer is too short and it is likely one or less words spoken which will not be picked up by VAD.

4.  Place config folder in `./icsi_data/`. Update config file to point to location of unprocessed data; should be: `./icsi_data/icsi_files/`.

5.  Run `data_preprocess.py` to create `train_tisv` and `test_tisv` folders. Train/test will automatically split 90/10. During script run, note output of speaker # showing numpy files containing 0 length arrays. 

6.  If there are speakers with 0 length arrays, find out which speaker name (folder) it belongs to by running `check_file.py`.

    i.  For ICSI Corpus:
    
        a. speaker10.npy (speaker folder: fe905) - TRAIN
        
        b. speaker35.npy (speaker folder: me908) - TRAIN
        
        c. speaker0.npy  (speaker folder: me907) - TEST
        
7.  Remove these speaker names (folders) from the unprocessed data location, and remove corresponding numpy speaker files from processed data location.
