import os
import sys
import librosa
import numpy as np

current_working_directory = os.getcwd() # get current working directory

# Address of root directory
base_path = current_working_directory + "/cap6610sp19_project/Test_Set/"
prog_path = base_path + "Prog/"
non_prog_path = base_path + "NonProg/"
djent_path = base_path + "djent/"

# Returns all audio files(.mp3, .avi, .wav) in the directory : path
def fileList(path) :
    matches = []
    for root, _, filenames in os.walk(path,topdown=True):
        for filename in filenames:
            if filename.endswith(('.mp3', '.wav', '.avi','.flac','.m4a','.ogg')):
                if matches.count(filename) == 0 :
                    matches.append(os.path.join(root, filename))
            
    return matches


# prog_files contains all prog_rock files
prog_files = fileList(prog_path)

# non_prog_files contains all non prog_rock files
non_prog_files = fileList(non_prog_path)

djent_files = fileList(djent_path)

# Returns all prog rock files
def get_prog_files():
    return prog_files


# Returns all non prog rock files
def get_non_prog_files() :
    return non_prog_files    

# Returns all djent files
def get_djent_files() :
    return djent_files    

