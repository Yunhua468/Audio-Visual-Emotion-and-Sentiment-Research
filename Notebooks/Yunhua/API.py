import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from keras.utils import np_utils

#Get the file names from the dataset
#eg: folder_path = 'C:/Users/ZhaoY/Downloads/DL_Project/dataset/Audio_Speech_Actors_01-24'
def get_file(folder_path):
    dir_list = os.listdir(folder_path)

    # get the paths
    dirts = []
    filepath_delete = True
    for root, dirt, file in os.walk(folder_path):
        if filepath_delete == True:
            print(root, "hahaha")
            filepath_delete = False
        else:
            dirts.append(root)
    # get the files in every path
    files = []
    for dirt in dirts:
        for root, dt, file in os.walk(dirt):
            for fl in file:
                files.append(os.path.join(dirt + fl))
    return files

def parse_filename(files):
    filename = []
    Actor = []
    for file in files:
        filename.append(file.split('\\')[-1])
        Actor.append(file.split('\\')[-2])
    Modality = []
    Vocal_channel = []
    Emotion = []
    Emotional_intensity = []
    Statement = []
    Repetition = []

    for name in filename:
        Modality.append(name.split('-')[0])
        Vocal_channel.append(name.split('-')[1])
        Emotion.append(int(name.split('-')[2])-1)
        Emotional_intensity.append(name.split('-')[3])
        Statement.append(name.split('-')[4])
        Repetition.append(name.split('-')[5])
    df = pd.DataFrame({'files': files,
                       'modalities': Modality,
                       'vocal_channels': Vocal_channel,
                       'emotions': Emotion,
                       'emotional_intensities': Emotional_intensity,
                       'statements': Statement,
                       'repetitiona': Repetition})
    return df,Emotion

# Get the feature vectors from mfcc
# mean_signal_length is the mean length of signals
def get_feature_vector_from_mfcc(signal, mean_signal_length: int, flatten: bool) -> np.ndarray:
    """
    Make feature vector from MFCC for the given wav file.

    Args:
        file_path (str): path to the .wav file that needs to be read.
        flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
        mfcc_len (int): Number of cepestral co efficients to be consider.

    Returns:
        numpy.ndarray: feature vector of the wav file made from mfcc.
    """
    # fs, signal = wav.read(file_path)
    # signal, fs = librosa.load(file_path, sr=16000, mono=True)
    s_len = len(signal)

    # pad the signals to have same size if lesser than required
    # else slice them

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    # sample/frame = mean_signal_length*frame_length
    mel_coefficients = mfcc(signal, fs, frame_length=0.048, frame_stride=0.024, num_filters=30, num_cepstral=30,
                            low_frequency=60, high_frequency=7600)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients