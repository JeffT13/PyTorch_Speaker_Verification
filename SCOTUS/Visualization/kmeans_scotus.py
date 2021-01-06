import os, json, pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
np.random.seed(13)



fp = '../../data/SCOTUS/'
case_path = fp+'dvectors/sco50d/'
    
total_cases = (len(os.listdir(case_path))/2)
train_cases = (total_cases//10)*9
print("# total cases:" , total_cases)
print("# of training:", train_cases)

trn_seq_lst = []
trn_cluster_lst = []
test_seq_lst = []
test_cluster_lst = []

verbose = True

if verbose:
    print("\n", "="*50, "\n Processing SVE d-vec")
        
#load 5 case-embedded dvecs (with directory holding raw files)
i = 0
for case in os.listdir(case_path):
    if case[-7:] == 'seq.npy':
        case_id = case.split('/')[-1].split('_')[0]

        train_sequence = np.load(case_path+case)
        train_clus = np.load(case_path+case_id+'_id.npy')

        if verbose:
            if i > train_cases:
                print("-- Stored as test case --")
            else:
                print("-- Stored as train case --")
            print('Processed case:', case_id)
            print('emb shape:', np.shape(train_sequence))
            print('label shape:', np.shape(train_clus))    

        #add to training or testing list (for multiple cases       
        if i < train_cases:
            trn_seq_lst.append(train_sequence)
            trn_cluster_lst.append(train_clus)
        else:
            test_seq_lst.append(train_sequence)
            test_cluster_lst.append(train_clus)
        i+=1
            
 

# Only Judge Embeddings
# Training & Test set Generation
judge_seq = []
judge_id = []
test_seq = []
test_id = []
for i, case in enumerate(trn_cluster_lst):
    case_seq = []
    case_id = []
    for j, emb in enumerate(case):
        if emb<20:
            case_seq.append(trn_seq_lst[i][j])
            case_id.append(emb)
    judge_seq.append(case_seq)
    judge_id.append(case_id)
            
        
for i, case in enumerate(test_cluster_lst):
    case_seq = []
    case_id = []
    for j, emb in enumerate(case):
        if emb<20:
            case_seq.append(test_seq_lst[i][j])
            case_id.append(emb)
    
    test_seq.append(case_seq)
    test_id.append(case_id)



limit = 30
X = np.concatenate([case for case in judge_seq[:limit]])
Y = np.concatenate([id for id in judge_id[:limit]])
num = len(np.unique(Y))
print('Number of speakers in training set:', num)

#1 test case
test = test_seq[0]
test_lab = test_id[0]


print('-- K-Means --')
print('-- Training --')
model = KMeans(n_clusters=num, random_state=0)
model.fit(X)
   
print('-- Inference --')
infer = model.predict(test)   

if verbose:
    print(len(infer), type(infer))
    print(len(test), type(test_lab)) 
    print('Visualize Prediction')
    print('='*50)
    print(test_lab[80:120])
    print(infer[80:120])

print('--- Centroid Array ---')
print(np.shape(model.cluster_centers_))

print('-- complete ---')