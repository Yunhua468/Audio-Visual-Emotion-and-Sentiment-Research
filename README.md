# Audio-Visual-Emotion-and-Sentiment-Research
Deep Neural Network and its application with TensorFlow project

#### Members:   
Audio parts: Enis Berk Çoban and Yunhua Zhao         
             We use the audio-song, audio-speech, audio-song-speech files separately to train models to detect emotion.  
             Also we use different models(VGGish and LSTM) to do the audio emotion recognition.  

Video parts: Patrick Jean-Baptiste and Tianyu Gao  
             We use images of actors' faces that express an emotion. The images are extracted from the video only and audio-visual files              of the RAVDESS dataset for both speech and song. The objective is to create a visual model to recognize emotions                        from images.
             
#### Period of our project:  
1) Explore the dataset: Enis extractes audio from video; Yunhua decode the filenames; Patrick extract images from the video files
2) Yunhua use LSTM to train models on audio-song files to get the accuracy; then use same model to the audio-speech files to get the      accuracy; then use same model to the audio-song-speech model.  
   Enis use VGGish to train the model on audio-song-speech files.
   Patrick does initial video preprocessing.
3) Enis split the dataset into train, validation, test sets, and make a csv file, so that everyone could use the same train-val-test dataset to compare the final result.  
4) Patrick detects and extracts the actors' faces from the images.
5) Enis try Yunhua's model to the splited dataset, Yunhua tried Enis' model; Patrick trains a model on the split dataset to do the visual emotion classification  
6) Enis proves several organized files and leads us to move on.
7) We make the PPT together.  

#### File organization:  
1) We use the "Issues" to track some problems  
2) Our email information: https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Documents/discussion_tracker.txt  
3) dataset info: https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Documents/Viki-Doc.txt  


