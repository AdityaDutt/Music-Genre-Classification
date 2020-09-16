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
import pandas as pd
import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Conv1D, Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
import random

prog = []
nonprog = []
total_files = 0
total_prog = 0
total_non_prog = 0

# find total files
for root, dirs, files in os.walk("/Users/richadutt/Documents/ranga/cap6610sp19_project/Test_Set/Prog", topdown=False):
   for name in files:
      filename = os.path.join(root, name)
      if filename.find(".mp3") != -1 or filename.find(".wav") != -1 or filename.find(".flac") != -1 or filename.find(".avi") != -1 or filename.find(".m4a") != -1 or filename.find(".ogg") != -1 :
          total_files = total_files + 1
          total_prog += 1 

for root, dirs, files in os.walk("/Users/richadutt/Documents/ranga/cap6610sp19_project/Test_Set/NonProg", topdown=False):
   for name in files:
      filename = os.path.join(root, name)
      if filename.find(".mp3") != -1 or filename.find(".wav") != -1 or filename.find(".flac") != -1 or filename.find(".avi") != -1 or filename.find(".m4a") != -1 or filename.find(".ogg") != -1 :
          total_files = total_files + 1
          total_non_prog += 1

# for root, dirs, files in os.walk("/Users/richadutt/Documents/ranga/cap6610sp19_project/Test_Set/djent", topdown=False):
#    for name in files:
#       filename = os.path.join(root, name)
#       if filename.find(".mp3") != -1 or filename.find(".wav") != -1 or filename.find(".flac") != -1 or filename.find(".avi") != -1 or filename.find(".m4a") != -1 or filename.find(".ogg") != -1 :
#           total_files = total_files + 1
#           total_prog += 1


# --------Extract data from csv file--------------------------------------

data = pd.read_csv('test_features.csv')
data.head()            
new_files = data['filename']
new_files = list(new_files)
for i in range(len(new_files)) :
    new_files[i] = (new_files[i].split("chunk") )[0]


data = data.drop(['filename'],axis=1)
data = np.asarray(data)

X = data[:,6:]
y = data[:,0]
y[y=="prog"] = 0
y[y=="non_prog"] = 1
y[y=="djent"] = 0

X = np.reshape(X, (X.shape[0],1, X.shape[1]))

print("X_train shape ",X.shape)
y_test = y
y_test = to_categorical(y_test)
 

model = load_model('Model.h5')

print("\nTesting ...")

# Predict using saved model

predictions = model.predict(X)
print(predictions)
row,col = predictions.shape
count = 0
for i in range(row) :
    true_val = np.argmax(y_test[i])
    predicted_val = np.argmax(predictions[i]) 
    print("true ",true_val,"    got ",predicted_val)
    if true_val == predicted_val :
       count += 1
print("Acc ",count," / ",row)   
print("Total files ",total_files)


new_true_ind = []
new_predicted_ind = []
i = 0

ind = []     
start = 0
for i in range(len(new_files)) :
    if i == 0 :
        start = 0
    elif i == len(new_files) - 1 :
        if new_files[i] == new_files[i-1] :
            ind.append([new_files[i],start,len(new_files)-1])  
        else :
            ind.append([new_files[i],len(new_files)-1,len(new_files)-1])  

    else :
         if new_files[i] != new_files[i-1] :
             end = i-1
             ind.append([new_files[i-1],start,end])  
             start = i
    
print("indices calculated") 

for i in range(row) :
    new_predicted_ind.append(np.argmax(predictions[i]))
    new_true_ind.append(np.argmax(y_test[i]))

# Find majority result for song chunks

def calc_majority(indices,predicted,true) :
    name = indices[0]
    start = indices[1]
    end = indices[2]
    new_predicted = predicted[start:end+1]
    new_true = true[start:end+1]
    maj_pred = 0
    maj_true = 0
    for i in range(len(new_predicted)) :
        if new_predicted[i] == 0 :
            maj_pred += 1
    if maj_pred <= 2*len(new_predicted)/3  :       
       ret_pred_val = 1
    else:
        ret_pred_val=0

    if name.find("nonprog") != -1 :
        ret_true_val = 1
    else :
        ret_true_val = 0        
    return [ret_pred_val,ret_true_val]



majority_pred = []
majority_true = []

print("len ind",len(ind))

# Find accuracy for all chunks by checking majority output
for i in range(len(ind)) :
    print(i)
    ret_pred_val,ret_true_val = calc_majority(ind[i],new_predicted_ind,new_true_ind)
    majority_pred.append(ret_pred_val)
    majority_true.append(ret_true_val)
count = 0
print("len maj",len(majority_pred))
for i in range(len(majority_pred)) :
    print("true ",majority_true[i],"     predicted",majority_pred[i])
    if majority_pred[i] == majority_true[i] :
        count += 1

print("Accuracy ",count," / ",total_files, (count/total_files) *100," %" )


confusion_matrix = confusion_matrix(majority_true, majority_pred ) 
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in ["True prog songs","True nonprog songs"]],
                  columns = [i for i in ["Predicted prog songs","Predicted nonprog songs"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.title('Total prog = %i, Total nonprog = %i' %(total_prog,total_non_prog) )
plt.show()