import os

path = "./librispeech/books/"
folders = os.listdir(path)

for book in folders:
    for speaker in os.listdir(path+book):
        for file in os.listdir(path+book+'/'+speaker):
            if '.wav' in file:
                if os.path.getsize(path+book+'/'+speaker+'/'+file)<40000:
                    os.remove(path+book+'/'+speaker+'/'+file)
                    os.remove(path+book+'/'+speaker+'/'+file.strip('.wav')+'.txt')