#####################################################################
#
# Description: Match Image File Names With Images
#
# This function retrieves images along with the corresponding labels 
# indicating the emotion expressed in the images. The main purpose
# is to create either a train, validation, or test set for visual 
# emotion.
#
# Input:
#   File_Names: An array containing the names of the image files. 
#   Images: An array containing the images of expressed emotion.
#   Labels: An array containing the labels indicating the emotion 
#           expressed in the images.
#   data_files: An array containing the image file name from a set
#               of data.
#
# Outputs:
#   X: A set of images of expressed emotion.
#   y_one_hot: A one-hot array containing the labels indicating the 
#              emotion expressed in the images.
#
# Author: Patrick Jean-Baptiste
#
#####################################################################


import numpy as np

def match_files(File_Names, Images, Labels, data_files):

    File_Names = File_Names.tolist()  
    X = []
    y = []    
    
    for idx, filename in enumerate(data_files):
        
        # Check if the image file name is in the array.
        if (File_Names.count(filename) > 0):
            
            # Retrieve the index that corresponds to the 
            # image file name in the array.
            index = File_Names.index(filename)
                
            # Append the image of an expressed emotion.
            X.append(Images[index])
            
            # Append the label indicating the type of emotion 
            # expressed to an array.
            y.append(Labels[index])
            
    
    X = np.array(X)
    y = np.array(y) 
    
    # Convert the labels into a one-hot array.
    y_one_hot = np.zeros((len(y), 8))

    for i in range(len(y)):
        y_one_hot[i,(y[i] - 1)] = 1 
    

    y_one_hot = y_one_hot.astype(int)
    
    return X, y_one_hot


