import os

path = "./librispeech/diarization"

for i in os.listdir(path):
    old_file = path + '/'+i
    new = i.strip('.ctm').replace('.','_')
    new_file = path + '/'+ new +'.ctm'
    os.rename(old_file,new_file)