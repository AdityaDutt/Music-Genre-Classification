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
from read_test_data import get_non_prog_files,get_prog_files,get_djent_files
sys.setrecursionlimit(10000)
djent_files = get_djent_files()

print("Number of djent songs",len(djent_files))

all_files = djent_files
fixed_sr = 44100
min_duration = 0
# -------------------------- Find minimum duration -------------------------------------------------

min_duration = 30#60.041
min_duration = int(min_duration)            
print("min duration ",min_duration)
# -------------------------- Feature extraction ----------------------------------------------------
file = open('test_djent_features.csv', 'w', newline='')
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

genre = 'djent'  
count = 0
time_series_length = 30
# Read prog files 
for i in range(len(all_files)) :
        print("djent ",i)
        filename = all_files[i]
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
                    
                    chroma_stft = librosa.feature.chroma_stft(y=chunk, sr=sr)
                    spec_cent = librosa.feature.spectral_centroid(y=chunk, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
                    rmse = librosa.feature.rmse(y=chunk)
                    rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)
                    zcr = 10**10*np.mean(librosa.zero_crossings(org_y)/len(org_y) )
                    
                    mfcc = librosa.feature.mfcc(y=chunk, sr=sr)
                    if np.mean(chroma_stft) == 0 :
                        break
                    to_append = f'{"djent"+name+"chunk"+str(chunk_index)} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    
                    
                    # Append all mfcc features i.e., 20 rows
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    
                    file = open('test_djent_features.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())

                    # spect = librosa.feature.melspectrogram(y=chunk, sr=sr,n_fft=2048, hop_length=512)
                    # spect = librosa.power_to_db(spect, ref=np.max)
                    # plt.figure(figsize=(14, 5))
                    # plt.axis('off')
                    # librosa.display.specshow(spect, fmax=8000) 

                    # plt.savefig("outputprog.png",bbox_inches='tight',transparent=True,pad_inches=0)
                    # im = cv2.imread("outputprog.png")
                    # plt.clf()
                    # plt.cla()
                    # plt.close()

                    # np.savez_compressed("/Users/richadutt/Documents/ranga/mfcc_validation_set/prog-"+name+"chunk"+str(chunk_index),im,im)
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
                to_append = f'{"djent"+name+"chunk1"} {genre} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {zcr}'    

                # Append all mfcc features i.e., 20 rows
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                
                file = open('test_djent_features.csv', 'a', newline='')
                with file:
                    writer = csv.writer(file)
                    writer.writerow(to_append.split())

                # spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
                # spect = librosa.power_to_db(spect, ref=np.max)

                # plt.figure(figsize=(14, 5))
                # plt.axis('off')
                # X = librosa.stft(y)
                # Xdb = librosa.amplitude_to_db(abs(X))
                # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 

                # plt.savefig("outputprog.png",bbox_inches='tight',transparent=True,pad_inches=0)
                # im = cv2.imread("outputprog.png")
                # plt.clf()
                # plt.cla()
                # plt.close()

                # np.savez_compressed("/Users/richadutt/Documents/ranga/mfcc_validation_set/prog-"+name+"chunk1",im,im)

            # print(i," ---->  ",y.shape)     


                
        except Exception as e :
            print("error handled")
            error_logs = open("error_logs.txt","a")
            error_logs.write(filename)
            error_logs.write("\n")
            error_logs.write(str(e))
            error_logs.write("\n")    
            error_logs.close()    
            continue