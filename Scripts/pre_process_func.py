import os
import soundfile as sf

import sys
from pathlib import Path
from time import gmtime, strftime

# sys.path.insert(0, './models/audioset')
from models.audioset.vggish_params import EXAMPLE_HOP_SECONDS
from models.audioset import vggish_input
# import vggish_postprocess

import shutil

import numpy as np

import math
import csv


def load_wav(input_file_path):
    """ Loads wav file and returns an array within wav data format

    waveform_to_examples is expecting `np.int16`,

    Args:
        input_file_path (str/Path): Path to the file is assumed to contain
             audio data.
    Returns:
        A tuple (wav_data, sampling_rate)
    """
    wav_data, sr = sf.read(input_file_path,dtype='int16')
    # print("wac_dat",wav_data.shape)
    return wav_data,sr

# this is complicated rather than formulated is that
# 41000/16000 is not an integer, but can be formulated TODO
# here is the calculation: given input size, calculates output seconds
# input_size=39975.0
# original_sr=41000
# sampling_ratio=16000/original_sr
# seconds=((input_size*sample_ratio)-(240))/(16000*0.96)
def cal_sample_size(wav_data,sr):
    """Cal. sample size from log mel spectogram for a given wav_file

        Args:
            wav_data (numpy.array): audio data in wav format
            sr (int): sampling rate of the audio

        Returns:
            [int,int,int]
    """
    assert (EXAMPLE_HOP_SECONDS==0.96)

    sampling_ratio=16000/sr
    lower_limit=((sr*0.96*1)+(240))/sampling_ratio
#     excerpt_limit=((sr*0.96*EXCERPT_LENGTH)+(240))/sampling_ratio

    offset=sr*EXCERPT_LENGTH
    #EXPECTED sample size, after processing
    sample_size=(len(wav_data)//offset)*EXCERPT_LENGTH
    remainder_wav_data=len(wav_data)%offset
    # if remainder will generate more than any samples (requires 42998 numbers)
    if remainder_wav_data<lower_limit:
        pass
    else:
        seconds=math.floor(((remainder_wav_data*sampling_ratio)-(240))/(16000*0.96))
        sample_size+=seconds

    return sample_size,offset,remainder_wav_data,lower_limit


def iterate_for_waveform_to_examples(wav_data,sr):
    """Wrapper for waveform_to_examples from models/audioset/vggish_input.py

        Iterate over data with 10 seconds batches, so waveform_to_examples produces
        stable results (equal size)
        read **(16/06/2019)** at Project_logs.md for explanations.

        Args:
            wav_data (numpy.array): audio data in wav format
            sr (int): sampling rate of the audio

        Returns:
            See waveform_to_examples.
    """
    sample_size,offset,remainder_wav_data,lower_limit=cal_sample_size(wav_data,sr)
    # in this loop wav_data jumps offset elements and sound jumps EXCERPT_LENGTH*2
    # because offset number of raw data turns into EXCERPT_LENGTH*2 pre-processed
    sound=np.zeros((sample_size,96,64),dtype=np.float32)
    count=0
    for i in range(0,len(wav_data),offset):
    #this is when wav_data%offset!=0
        # numpy indexing handles bigger indexes
        # i+offset>len(wav_data) means that we are on the last loop
        # then if there is enough remaind data, process it otherwise not
        if i+offset>len(wav_data) and remainder_wav_data<lower_limit:
            continue
        # left data is smaller than 22712, we cannot pre-process
        # if smaller than 42998, will be 0 anyway
        a_sound= vggish_input.waveform_to_examples(wav_data[i:i+(offset)], sr)
        sound[count:(count+a_sound.shape[0]),:,:]=a_sound[:,:,:]
        count+=a_sound.shape[0]
    return sound


def mp3file_to_examples(mp3_file_path):
    """Wrapper around iterate_for_waveform_to_examples() for a common mp3 format.

    Args:
        mp3_file_path (str/Path): String path to a file. The file is assumed to contain
            mp3 audio data.

    Returns:
        See iterate_for_waveform_to_examples.
    """
    extension=Path(mp3_file_path).suffix

    if extension.lower()==".wav":
        wav_data,sr=load_wav(mp3_file_path)
    else:
        print("ERROR file extension {} is not supported.".format(extension))
        return None
    sys.stdout.flush()
    sys.stderr.flush()

    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    #######iterate over 10 seconds#########
    sound = iterate_for_waveform_to_examples(samples,sr)
    return sound

def pre_process(mp3_file_path,output_dir="./", saveAsFile=False):
    """Wrapper for mp3file_to_examples, handles input and output logic

        Saves as a file called mp3_file_name_preprocessed.npy in output_dir
        If output npy file already exists returns None

        Args:
            mp3_file_path (numpy.array): audio data in wav format
            output_dir (str/Path): output directory
            saveAsFile (bool): save as file or not
            sr (int): sampling rate of the audio

        Returns:
            Returns pre_processed sound (numpy.array,np.float32) if file does not exists
    """
    mp3_file_path = Path(mp3_file_path)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        os.mkdir(output_dir)

    npy_file_path = Path(output_dir) / (str(mp3_file_path.stem) + "_preprocessed.npy")

    if npy_file_path.exists():
        return None
    sound = mp3file_to_examples(mp3_file_path)
    sound = sound.astype(np.float32)

    if saveAsFile:
        np.save(npy_file_path,sound)
    return sound
