##################################################################
#
# Description: Face Detection & Extraction
#
# This process consists of detecting the speaker's face in a given 
# grayscale video frame. Then, extracting the face from the video 
# frame.
#
#
# Inputs:
#   gray_frame: A grayscale frame from a video.
#   face_cascade: A face detector.
#   img_size: Size of the output image containing the extracted 
#             face from the video frame.
#
# Output:
#   extracted: An image containing the extracted face from the 
#              video frame.
#
# Author: Patrick Jean-Baptiste
#
##################################################################


import cv2

def detection_and_extraction(gray_frame, face_cascade, img_size):
   
    # Detect the face in the video frame.
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 4)

    # Extract the face from the video frame.
    for (x, y, w, h) in faces:
        extracted = gray_frame[y:(y + h), x:(x + w)] 
        
        
    # Reshape the image of the extracted face.
    extracted = cv2.resize(extracted, (img_size, img_size))

    return extracted


