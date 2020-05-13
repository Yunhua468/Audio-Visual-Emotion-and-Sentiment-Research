#####################################################################
#
# Description: Get Images, Labels, and File Names
#
# This function retrieves images along with the names of the image 
# files and the corresponding labels indicating the emotion expressed
# in the images.
#
# Input:
#   img_dir: A directory containing the images of expressed emotion.
#
# Outputs:
#   Images: An array containing the images of expressed emotion.
#   File_Names: An array containing the names of the image files. 
#   Labels: An array containing the labels indicating the emotion 
#           expressed in the images.
#   img_size: The size of an image in the directory.
#
# Author: Patrick Jean-Baptiste
#
#####################################################################


import cv2
import os
import numpy as np

def get_images(img_dir):

    Images = []
    Labels = []
    File_Names = []
    
    for idx, img_file in enumerate(os.listdir(img_dir)):
        
        # Read an image of an actor's face expressing an emotion.
        img = cv2.imread(os.path.join(img_dir, img_file))
        
        # Append the image of an expressed emotion to an array.
        Images.append(img)
        
        # Append the label indicating the type of emotion 
        # expressed to an array.
        Labels.append(int(img_file[7]))
               
        # Append the name of the file that contains the image.
        File_Names.append(img_file)
        
        
    Images = np.array(Images)  
    Labels = np.array(Labels)
    File_Names = np.array(File_Names)
    img_size = Images.shape[1]
    
    return Images, Labels, File_Names, img_size


