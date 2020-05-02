import os
import pandas as pd

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
        Emotion.append(name.split('-')[2])
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
    return df