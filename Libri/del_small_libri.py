import os

rootDir = "./librispeech/speakers/"

def list_files(filepath, filetype):
#    paths = []
    for root, dirs, files in os.walk(filepath):
        count = 0
        for file in files:
            
            wav_path = os.path.join(root, file)
            txt_path = wav_path.strip('wav')+'txt'
            
            if file.lower().endswith(filetype.lower()) and os.path.getsize(wav_path)<40000:
                
                print('wav exists:',os.path.isfile(wav_path))
                print('txt exists:',os.path.isfile(wav_path.strip(wav)+'txt')
                
                os.remove(wav_path)
                os.remove(txt_path)
                
                count += 1
    print('Deleted {} wav files'.format(count))
                      
list_files(rootDir,'.wav')
