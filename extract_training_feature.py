import os
import csv
import sys
import math
import cv2
import scipy
import pickle
import librosa
import matplotlib
import numpy as np
import librosa.display
import IPython.display as ipd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# print(os.getcwd())
# sys.exit(1)
# Import other project files
from read_input import get_non_prog_files,get_prog_files
sys.setrecursionlimit(10000)
prog_files = get_prog_files()
non_prog_files = get_non_prog_files()

print("Number of prog songs",len(prog_files))
print("Number of non prog songs",len(non_prog_files))

all_files = prog_files + non_prog_files
fixed_sr = 44100
min_duration = 0
# -------------------------- Find minimum duration -------------------------------------------------

for i in range(len(all_files)) :
        filename = all_files[i]
        y, sr = librosa.load(filename,sr = fixed_sr)     
        curr_duration = librosa.get_duration(y=y, sr=sr)
        if i == 0 :
            min_duration = curr_duration
        else :
            min_duration = min(min_duration,curr_duration)    
#min_duration = 30#60.041
min_duration = int(min_duration)            
print("min duration ",min_duration)
# -------------------------- Feature extraction ----------------------------------------------------
file = open('training_features.csv', 'w', newline='')
header = 'filename genre chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):
    header += f' mfcc{i}'

header = header.split()

# Create file to write error logs
error_logs = open("error_logs.txt","w")
error_logs.close()    
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genre = 'prog'  
count = 0
time_series_length = 30
# Read prog files 
for i in range(len(prog_files)) :
        print("prog ",i)
        filename = prog_files[i]
        name = (filename.split("/") )[-1]
        name = name.replace(" ","_")

        try:
            y, sr = librosa.load(filename,sr = fixed_sr) 
            time = librosa.get_duration(y=y,sr=sr)
            chunks = []
            if time > min_duration :
                org_y = y
                iter = math.floor(time/min_duration)
                print(iter)

                current_size = time*fixed_sr
                chunk_size = min_duration*fixed_sr
                start = 0#math.floor(iter/3) * chunk_size
                end = chunk_size
                chunk_index = 1
                #iter = math.floor(iter/3)

                while iter !=0 :
                    count += 1
                    chunk = y[start:end]
                    
                    chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr)
                    spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
                    rmse = librosa.feature.rmse(y=chunk)
                    rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
                    zcr = 10**10*np.mean(librosa.zero_crossings(org_y)/len(org_y) )
                    mfcc = librosa.feature.mfcc(y=chunk, sr=sr)
                    to_append = f'{"prog"+name+"chunk"+str(chunk_index)} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    
                    
                    # Append all mfcc features i.e., 20 rows
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    
                    file = open('training_features.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

                    # spect = librosa.feature.melspectrogram(y=chunk, sr=sr,n_fft=2048, hop_length=512)
                    # spect = librosa.power_to_db(spect, ref=np.max)
                    # plt.figure(figsize=(14, 5))
                    # plt.axis('off')
                    # librosa.display.specshow(spect, fmax=8000) 
                    # plt.show()

                    chunk_index += 1


                    start = end
                    end = end + chunk_size
                    iter -= 1
                    # if chunk_index >=20 :
                    #     break
                # print("chunk size ",chunk_size) 
                # print("current song size ",current_size)   
            else :
                count += 1
                y, sr = librosa.load(filename,sr = fixed_sr,duration=min_duration) 

                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rmse = librosa.feature.rmse(y=y)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = 10**10*np.mean(librosa.zero_crossings(org_y)/len(org_y) )
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{"prog"+name+"chunk1"} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    

                # Append all mfcc features i.e., 20 rows
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                
                file = open('training_features.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

                # spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
                # spect = librosa.power_to_db(spect, ref=np.max)
                # plt.figure(figsize=(14, 5))
                # plt.axis('off')
                # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 
                # plt.show()


                
        except Exception as e :
            print("error handled")
            error_logs = open("error_logs.txt","a")
            error_logs.write(filename)
            error_logs.write("\n")
            error_logs.write(str(e))
            error_logs.write("\n")    
            error_logs.close()    
            continue


genre = 'non_prog'   

# Read Non prog files
for i in range(len(non_prog_files)) :
        print("nonprog ",i)
        filename = non_prog_files[i]
        name = (filename.split("/") )[-1]
        name = name.replace(" ","_")


        try:
            y, sr = librosa.load(filename,sr = fixed_sr) 
            time = librosa.get_duration(y=y,sr=sr)
            chunks = []
            if time > min_duration :
                org_y = y
                iter = math.floor(time/min_duration)
                print(iter)

                current_size = time*fixed_sr
                chunk_size = min_duration*fixed_sr
                start = 0#math.floor(iter/3) * chunk_size
                end = chunk_size
                chunk_index = 1
                # iter = math.floor(iter/3)

                while iter !=0 :
                    count += 1
                    chunk = y[start:end]
                    
                    # Display Spectrogram
                    # spect = librosa.feature.melspectrogram(y=chunk, sr=sr,n_fft=2048, hop_length=512)
                    # spect = librosa.power_to_db(spect, ref=np.max)
                    # plt.figure(figsize=(14, 5))
                    # plt.axis('off')
                    # plt.show()
                    
                    chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr)
                    spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
                    rmse = librosa.feature.rmse(y=chunk)
                    rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
                    zcr = 10**10*np.mean(librosa.zero_crossings(org_y)/len(org_y) )
                    mfcc = librosa.feature.mfcc(y=chunk, sr=sr)
                    to_append = f'{"nonprog"+name+"chunk"+str(chunk_index)} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    

                    # Append all mfcc features i.e., 20 rows
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    
                    file = open('training_features.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

                    chunk_index += 1

                    # print("start ",start)
                    # print("end ",end)

                    start = end
                    end = end + chunk_size
                    iter -= 1
                    # if chunk_index >=20 :
                    #     break

            else :
                count += 1
                y, sr = librosa.load(filename,sr = fixed_sr,duration=min_duration) 

                # Display Spectrogram
                # spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
                # spect = librosa.power_to_db(spect, ref=np.max)
                # plt.figure(figsize=(14, 5))
                # plt.axis('off')
                # librosa.display.specshow(spect, sr=sr, x_axis='time', y_axis='hz') 
                # plt.show()

                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rmse = librosa.feature.rmse(y=y)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = 10**10*np.mean(librosa.zero_crossings(org_y)/len(org_y) )
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{"nonprog"+name+"chunk1"} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    

                # Append all mfcc features i.e., 20 rows
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                
                file = open('training_features.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())


        except Exception as e:
            print("error handled")
            error_logs = open("error_logs.txt","a")
            error_logs.write(filename)
            error_logs.write("\n")
            error_logs.write(str(e))
            error_logs.write("\n")    
            error_logs.close()    
            continue
print("tot count ",count)