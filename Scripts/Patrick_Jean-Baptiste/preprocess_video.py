##################################################################
#
# Description: Video Preprocessing
#
# Performs the initial video preprocessing which involves
# reading a video file, extracting video frames, and converting
# the video frames to grayscale.
#
#
# Input:
#   vid_file: Name of a video file.
#
# Output:
#   gray_frame: A frame from the video converted to grayscale.
#
# Author: Patrick Jean-Baptiste
#
##################################################################


import cv2

def preprocess_video(vid_file):
   
    # Capture the video from the video file.
    cap = cv2.VideoCapture(vid_file)

    # Get the total number of frames in the video.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Retrieve a frame in the middle of the video.
    cap.set(1, (int(total_frames / 2) - 1));
    
    # Read the video frame.
    ret, frame = cap.read()

    # Convert the video frame from RGB to grayscale.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cap.release()
            
    return gray_frame        
            

