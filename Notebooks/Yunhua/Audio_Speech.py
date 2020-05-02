from API import get_file, parse_filename, get_feature_vector_from_mfcc
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
from sklearn.utils.multiclass import unique_labels

FILEPATH_AUDIO_SPEECH = 'C:/Users/ZhaoY/Downloads/DL_Project/dataset/Audio_Speech_Actors_01-24'
audio_speech_file = get_file(FILEPATH_AUDIO_SPEECH)
print(len(audio_speech_file))

FILEPATH_AUDIO_SONG = 'C:/Users/ZhaoY/Downloads/DL_Project/dataset/Audio_Song_Actors_01-24'
audio_song_file = get_file(FILEPATH_AUDIO_SONG)
print(len(audio_song_file))

#mix the song and speech files together into list 'files'
files = []
for file in audio_speech_file:
    files.append(file)
for file in audio_song_file:
    files.append(file)

df, emotions = parse_filename(files)
print(df)

# get signal and the mean length of signals from the .wav files
def get_signal_mean_length(file_path):
    mean_signal_length = 0
    signals = []
    for fname in file_path:
        signal, fs = librosa.load(fname, sr=16000, mono=True)
        mean_signal_length += len(signal)
        signals.append(signal)
    mean_signal_length = int(mean_signal_length / (len(files)))
    return signals, mean_signal_length

signal, mean_signal_length = get_signal_mean_length(files)

# extract features from mfcc
features = []
for sigl in signal:
    features.append(get_feature_vector_from_mfcc(sigl, mean_signal_length, flatten=False))

# encode the emotion lables and adjust features to the model
labels = np_utils.to_categorical(Emotion)
features = np.vstack([feature[np.newaxis, :, :] for feature in features])

# split data set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# define and compile model
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=(input_shape[0], input_shape[1])))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='tanh'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = build_model(input_shape=(features.shape[1], features.shape[2]), num_classes=labels.shape[1])
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.2,
                               patience=5,
                               min_lr=1e-6,
                               verbose=1)
hist = model.fit(x_train, y_train, batch_size=32, epochs=34, validation_data=(x_test, y_test), callbacks=[lr_reducer])

print("done")

