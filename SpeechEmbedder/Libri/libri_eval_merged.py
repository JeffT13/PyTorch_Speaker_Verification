import os
from pydub import AudioSegment

'''Creates structure: librispeech/books/speakers/speaker_id'''
'''books folder only contains speakers invovled in the specific book reading'''
'''Speaker durations are merged for the entire time they are speaking.'''

path = "./librispeech/diarization"

diarization = os.listdir(path)

#check files 
files = []
for file in diarization:
    files.append(file)

#create dir for speaker folders
os.makedirs('./librispeech/books')

book_path = './librispeech/books/'

s = 1
for file in files:
    f = open(path+'/'+file, "r")

    data = []
    count = 0 #to keep track of line number
    for line in f:
        data.append(line.split(' '))

        #convert times from string to float
        data[count][2] = float(data[count][2])
        data[count][3] = float(data[count][3])

        #duration is recorded; add start time to get end time
        data[count][3] += data[count][2]
        data[count][3] = round(data[count][3],2)

        count+=1

    #remove durations where words=1 
    times = []
    for dat in data:
        if dat[4] != '1\n':
            times.append(dat)

    #####
    #loop below merges speaking durations by speaker
    merged_times = []

    #initialize current speaker, start time, end time of speaking

    start_time = times[0][2]
    end_time = times[0][3]

    #iterate through each speaking duration
    for i in range(len(times)-1):
        curr = times[i][0]
        #if current speaker is same as next speaker
        if curr == times[i+1][0]:

            #check if speaking duration is continuous
            if times[i][3] == times[i+1][2]:

                #if it is, set as new end time of duration
                end_time = times[i+1][3]

            #if not, end the speaking duration by saving the last speaker's speaking duration
            else:
                merged_times.append(times[i][0]+' '+str(start_time)+' '+str(end_time))
                start_time = times[i+1][2]
                end_time = times[i+1][3]

        #if there is a new speaker
        else:

            #save the last speaker's speaking duration
            merged_times.append(times[i][0]+' '+str(start_time)+' '+str(end_time))
            start_time = times[i+1][2]
            end_time = times[i+1][3]
    
    #save last speaker's speaking duration
    merged_times.append(times[i][0]+' '+str(start_time)+' '+str(end_time))
    ####

    ###
    start_time = merged_times[0].split(' ')[1]
    end_time = merged_times[0].split(' ')[2]

    new_ls = []

    for t in range(len(merged_times)-1):
        curr = merged_times[t].split(' ')[0]
        if curr == merged_times[t+1].split(' ')[0]:
            end_time = merged_times[t+1].split(' ')[2]

        else:
            new_ls.append(curr+' '+start_time+' '+end_time)
            start_time = merged_times[t+1].split(' ')[1]
            end_time = merged_times[t+1].split(' ')[2]

    new_ls.append(curr+' '+start_time+' '+end_time)

    os.makedirs(book_path+'Book_'+str(s))
    speak_path = book_path+'Book_'+str(s)
    
    for t in new_ls:
        speaker = t.split(' ')[0].split('-')[0]+'-'+t.split(' ')[0].split('-')[1]
        #if no speaker folder, create it
        if speaker not in os.listdir(speak_path):
            os.makedirs(speak_path+'/'+speaker)

        #create file number 
        file_num = len(os.listdir(speak_path+'/'+speaker))+1

        #save start time, end time, transcript as .txt file in speaker folder
        with open(speak_path+'/'+speaker+'/'+file.strip('.ctm')+'_{}.txt'.format(file_num),'w') as output:
            output.write(str(t.split(' ')[1])+' '+ str(t.split(' ')[2]))

        #open corresponding audio file
        audio_file = './librispeech/audio/'+file.strip('.ctm')+'.wav'
        audio = AudioSegment.from_wav(audio_file)

        #split audio file based on start, end times
        start = float(t.split(' ')[1])*1000
        end = float(t.split(' ')[2])*1000
        split = audio[start:end]

        #save audio file in speaker folder with same file number
        split.export(speak_path+'/'+speaker+'/'+file.strip('.ctm')+'_{}.wav'.format(file_num),format='wav')

    print('Done ' + file)

    #increment folder num
    s+=1