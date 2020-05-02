from API import get_file, parse_filename
import numpy as np

FILEPATH_AUDIO_SPEECH = 'C:/Users/ZhaoY/Downloads/DL_Project/dataset/Audio_Speech_Actors_01-24'
audio_speech_file = get_file(FILEPATH_AUDIO_SPEECH)
print(len(audio_speech_file))

# FILEPATH_AUDIO_SONG = 'C:/Users/ZhaoY/Downloads/DL_Project/dataset/Audio_Song_Actors_01-24'
# audio_song_file = get_file(FILEPATH_AUDIO_SONG)
# print(len(audio_song_file))


df = parse_filename(audio_speech_file)
