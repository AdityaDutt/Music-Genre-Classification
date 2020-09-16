# Music Genre Classification

Clasify music into two categores progessive rock and non-prog rock.

Firstly, minimum duration song, min_duration was found. Then, all songs were divided into chunks of minimum of 30 seconds or min_duration. Librosa is used to analyze music extract frequencies. It is a Python module to analyze audio signals in general but geared more towards music. It includes the nuts and bolts to build a MIR (Music information retrieval) system. After building features, two approaches were considered to build genre classifier 
- Extract a Mel spectrogram of song chunk and then design a convolution neural net to run on input spectrograms. Spectrograms of a prog and non prog song are shown in Fig. 1 and Fig 2. The accuracy with spectrograms was 70-73%.
- Extract 21 Mfcc features, zero crossing rate, chroma frequencies, spectral bandwidth, spectral centroid, roll off for each chunk. Then, all the features were appended into csv file using pandas. Create a model that uses LSTM with 2 layers and runs on input features. The accuracy with LSTM was 80-85%. An image of features is shown in Fig. 2.

- Fig1. Spectrogram of non-prog rock music chunk
<img align="left" alt="non-prog" src="/non-prog.png" />

- Fig1. Spectrgotam of prog rock music chunk
<img align="left" alt="prog"  src="/prog.png" />


# Steps to run the project - 

1. To train the model run- python LSTM.py
2. TO validate model run- python validate_model.py
3. To test model run- python test_model.py

# Files 

All the training features are stored in training_features.py
All the validation features are stored in validation_features.py
All the test features are stored in test_features.py
All the test djent features are stored in test_djent_features.py


## Group Members

1. Aditya Dutt
2. Richa Dutt
3. DingKang Wang
4. Bin XU
5. Kun Shi


