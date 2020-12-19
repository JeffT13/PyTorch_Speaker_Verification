# TIMIT Speaker Verification

**README from [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification)**
**dvector scripts & `src` based on repo**

## Preprocessing

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
