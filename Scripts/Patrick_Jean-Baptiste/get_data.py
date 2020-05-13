##################################################################
#
# Description: Get Image File Names and Labels
#
# This function retrieves the names of the image files along with 
# the corresponding labels indicating the emotion expressed in 
# the images.
#
# Input:
#   df: A data frame.
#
# Outputs:
#   Img_Files: An array containing the names of the image files. 
#   Img_Labels: An array containing the labels indicating the 
#               emotion expressed in the images.
#
# Author: Patrick Jean-Baptiste
#
##################################################################


import numpy as np

def get_data(df):
    
    Img_Files = []
    Img_Labels = []
    
    for row in df.iterrows():
        
        img_file = row[1][0].split(".")[0] + ".jpg" 
    
        img_file1 = img_file.replace(img_file[0:2], "01")
        img_file2 = img_file.replace(img_file[0:2], "02")
        
        Img_Files.append(img_file1)
        Img_Files.append(img_file2)
        
        if (row[1][1] == "neutral"):
            Img_Labels.append('1')
            Img_Labels.append('1')
            
        elif (row[1][1] == "calm"):
            Img_Labels.append('2')
            Img_Labels.append('2')
            
        elif (row[1][1] == "happy"):
            Img_Labels.append('3')
            Img_Labels.append('3')
            
        elif (row[1][1] == "sad"):
            Img_Labels.append('4')
            Img_Labels.append('4')
            
        elif (row[1][1] == "angry"):
            Img_Labels.append('5')
            Img_Labels.append('5')
            
        elif (row[1][1] == "fear"):
            Img_Labels.append('6')
            Img_Labels.append('6')
            
        elif (row[1][1] == "disgust"):
            Img_Labels.append('7')
            Img_Labels.append('7')
            
        elif (row[1][1] == "surprise"):
            Img_Labels.append('8')
            Img_Labels.append('8')
            
     
    Img_Files = np.array(Img_Files)
    Img_Labels = np.array(Img_Labels)
    
    return Img_Files, Img_Labels
    
            
    