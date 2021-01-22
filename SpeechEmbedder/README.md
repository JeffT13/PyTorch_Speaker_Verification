# TIMIT Speaker Verification

**README from [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification)**

PyTorch implementation of speech embedding net and loss described here: https://arxiv.org/pdf/1710.10467.pdf.

Also contains code to create embeddings compatible as input for the speaker diarization model found at https://github.com/google/uis-rnn

![training loss](https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/Results/Loss.png)

The TIMIT speech corpus was used to train the model, found here: https://catalog.ldc.upenn.edu/LDC93S1,
or here, https://github.com/philipperemy/timit

# Preprocessing

Change the following config.yaml key to a regex containing all .WAV files in your downloaded TIMIT dataset. The TIMIT .WAV files must be converted to the standard format (RIFF) for the dvector_create.py script, but not for training the neural network.
```yaml
unprocessed_data: './TIMIT/*/*/*/*.wav'
```
Run the preprocessing script:
```
./data_preprocess.py 
```
Two folders will be created, train_tisv and test_tisv, containing .npy files containing numpy ndarrays of speaker utterances with a 90%/10% training/testing split.

# Training

To train the speaker verification model, run:
```
./train_speech_embedder.py 
```
with the following config.yaml key set to true:
```yaml
training: !!bool "true"
```
for testing, set the key value to:
```yaml
training: !!bool "false"
```
The log file and checkpoint save locations are controlled by the following values:
```yaml
log_file: './speech_id_checkpoint/Stats'
checkpoint_dir: './speech_id_checkpoint'
```
Only TI-SV is implemented.

# Performance

| Left-Aligned  | Center Aligned  | Right Aligned |
| Data          | # Speakers      | EER   |
| :------------ |:---------------:| -----:|
| TIMIT      | 630 | $1600 | .0491 |
| TIMIT+ICSI      | 689       |   .0764 |
| TIMIT+ICSI+Libri | 1274        |    .08 |

# D vector embedding creation

Follow the README in  `../SCOTUS` for d-vector generation procedure (with and without speaker labels). 
