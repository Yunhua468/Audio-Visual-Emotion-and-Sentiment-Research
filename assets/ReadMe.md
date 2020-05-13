#### This folder will be used to store required files to run experiments such as model checkpoints.

* Vggish embeddings for audio files:
[dropboxLink](https://www.dropbox.com/s/yigijs122togfk4/embeddings.dat?dl=1)

    * ```python3
    import pickle
    with open('embeddings.dat', 'rb') as file:
        embeddings=pickle.load(file)
    ```
    * embeddings is a dictionary with two keys "raw" and "post". "post" represents embedings that are PCA and whitening applied as well as quantization to 8 bits.
    Values corresponding to "raw" and "post" are dictionaries, keys are file names and values are numpy arrays consists of embeddings.
    Each embedding has a size of [n,128], n representing number of seconds in the audio file.
