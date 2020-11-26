# SCOTUS Speaker Verification

## SCOTUS Preprocessing

- Oyez
- PRINCE


# d-vector embedding

This procedure outputs a folder for each case processed, each containing a numpy array of dvector embeddings (`case_sequence.npy`) and a numpy array of labels (`case_cluster_id.npy`) which are both the same length, as well as a csv of a list of files which were not embedded (usually do to being too short). These are formatted for [our fork of the uisrnn](https://github.com/JeffT13/uis-rnn) 
