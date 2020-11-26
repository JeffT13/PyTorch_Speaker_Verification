import os

audio_path = "./librispeech/audio"

for i in os.listdir(audio_path):
    old_file = audio_path + '/'+i
    new = i.strip('.wav').replace('.','_')
    new_file = audio_path + '/'+ new +'.wav'
    os.rename(old_file,new_file)