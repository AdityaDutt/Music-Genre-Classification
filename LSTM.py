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
from keras.layers import Dense, Conv2D,Conv1D, Flatten,Dropout,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import KFold

# split data into test and train
def split_data_test_train(prog,nonprog) :
    ind1 = math.floor(len(nonprog)/3)
    ind2 = math.floor(len(prog)/3)
    x_train1 = nonprog[0:ind1]
    y_train1 = np.ones((ind1,1 ))
    x_test1 = nonprog[ind1:]
    y_test1 = np.ones((len(nonprog) - ind1,1) )
    print(len(x_test1) )
    print(len(y_test1) )

    x_train2 = prog[0:ind2]
    y_train2 = np.zeros( ( ind2,1 ))
    x_test2 = prog[ind2:]
    y_test2 = np.zeros( (len(prog) - ind2,1 ))

    print(len(x_test2) )
    print(len(y_test2) )
    
    
    x_test = x_train1 + x_train2
    x_train = x_test1 + x_test2 
    y_test = np.concatenate((y_train1, y_train2), axis=0)
    y_train = np.concatenate((y_test1, y_test2), axis=0)
    print("x_train ",len(x_train)  )
    print("x-test ",len(x_test)  )
    print("y_train ",len(y_train)  )
    print("y_test ",len(y_test)  )
    return [x_train,x_test,y_train,y_test]


# Get train data from csv file
x_train = []
y_train = []
prog = []

nonprog = []

data = pd.read_csv('training_features.csv')
data.head()            
data = data.drop(['filename'],axis=1)
data = np.asarray(data)
X = data[:,6:]
y = data[:,0]
y[y=="prog"] = 0
y[y=="non_prog"] = 1

x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=42)

X = np.reshape(X, (X.shape[0],1,X.shape[1]))
    

print("X_train shape ",x_train.shape)
print("X_test shape ",y_train.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y = to_categorical(y)

# Build model using LSTM

input_shape = (1,X.shape[2])
kfold = KFold(n_splits=3, shuffle=True, random_state=45)
cvscores = []
batch_size = 35
nb_epochs = 100
opt = Adam()

count = 1
for train, test in kfold.split(X, y):
  # Create model
    model = Sequential()
    model.add(LSTM(units=64, dropout=0.01, recurrent_dropout=0.35, return_sequences=True,input_shape=input_shape) )
    model.add(LSTM(units=32, dropout=0.01, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=2, activation='sigmoid'))

    print("Compiling ...")
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    print("Training ...")
    history = model.fit(X[train], y[train], batch_size=batch_size, epochs=nb_epochs,validation_split=0.33)
    score, accuracy = model.evaluate(X[test], y[test], batch_size=batch_size, verbose=1)
    print("Test loss:  ", score)
    print("Test accuracy:  ", accuracy)

    # Summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("/Users/richadutt/Documents/ranga/Graphs1/Accuracy - epoch"+str(count))
    plt.close()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("/Users/richadutt/Documents/ranga/Graphs1/Loss - epoch"+str(count))
    plt.close()

    count += 1
model.save('Model.h5')