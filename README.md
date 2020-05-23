
# Audio-Visual-Emotion-and-Sentiment-Research
Deep Neural Network and its application with TensorFlow project

#### Members and Corresponding parts:   
Audio parts: **Enis Berk Çoban** and **Yunhua Zhao**    

We use the audio-song, audio-speech, audio-song-speech files separately to train models to detect emotion. Also we use different NN architectures (LSTM, CNNs) along with pre-trained models for transfer learning (VGGish).

Video parts: **Patrick Jean-Baptiste**

We use images of actors' faces that express an emotion. The images are extracted from the video only and audio-visual files of the RAVDESS dataset for both speech and song. The objective is to create a visual model to recognize emotions from images.
             
#### Period of our project:  

1. Explore and pre-process the dataset:\
&ensp;Enis extractes audio from video;\
&ensp;Yunhua decode the filenames;\
&ensp;Patrick extract images from the video files
2. \
&ensp;Yunhua use LSTM to train models on audio-song files to get the accuracy; then use same model to the audio-speech files to get the accuracy; then use same model to the audio-song-speech model.\
&ensp;Enis generated VGGish embeddings to train the model on audio-song-speech files.\
&ensp;Patrick does initial video preprocessing.
3. \
&ensp;Enis split the dataset into train, validation, test sets, and made a csv file, so that everyone could use the same sets for training and we can merge models or their outputs.\
4. \
&ensp;Patrick detects and extracts the actors' faces from the images.
5. \
&ensp;Enis tried reproducing Yunhua's results and discovered a bug,\
&ensp;Yunhua tried Enis' model output with Yunhua's model;\
&ensp;Patrick trains a model on the split dataset to do the visual emotion classification  
6. \
&ensp;Enis provided several organized files and leads us to move on.
7. \
&ensp;Enis trained a model which has a module for each type of input.
&ensp;Yunhua merged the original features of audio and video and put the merged features to one dense layers model.  
9. We prepared the presentation.  

#### File organization:  
1. We use the "Issues" to track some problems  such as
	* [Creating dataset splits](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/issues/3)
	* [How to merge audio and video features?](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/issues/4)
2. Everybody completed their experiments on notebooks such as:
	* [Training Deep models on VGGish embeddings](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Notebooks/EnisBerk/DeepModelOnEmbeds.ipynb) by [@EnisBerk](https://github.com/EnisBerk)
	* [LSTM for sound features ](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Notebooks/Yunhua/DL_project_audio_speach_song.ipynb) by [@Yunhua468](https://github.com/Yunhua468)
	* [Emotion detection from Images](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Notebooks/Patrick_Jean-Baptiste/visual_emotion_recognition.ipynb) by [@patrick-jean-baptiste](https://github.com/patrick-jean-baptiste)
3. We created some scripts that handles data processing functions such as:
	* [VGGish model API](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Scripts/models_api.py) by [@EnisBerk](https://github.com/EnisBerk)
	* [Loading Files](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Notebooks/Yunhua/API.py)  by [@Yunhua468](https://github.com/Yunhua468)
	* [Detecting and Extracting faces](https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Scripts/Patrick_Jean-Baptiste/detection_and_extraction.py) by  [@patrick-jean-baptiste](https://github.com/patrick-jean-baptiste)
4. Our email information: https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Documents/discussion_tracker.txt  
5. dataset info: https://github.com/Yunhua468/Audio-Visual-Emotion-and-Sentiment-Research/blob/master/Documents/Viki-Doc.txt  



