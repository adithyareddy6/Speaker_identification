# Speaker_identification
i took almost 1500 sec of audio of each person and trained model to identify pattens specific to each person and recognize the person.


**Speaker Recognition System
This repository contains a speaker recognition system built using a convolutional neural
network (CNN) with residual blocks. The system is trained on a dataset of audio recordings
and can be used to identify different speakers based on their voice.**
## Installation
To run the code, you need to have the following installed:
* Python 3.6 or higher
* TensorFlow 2.0 or higher
* Librosa
* NumPy
* Scikit-learn
Trained model: https://drive.google.com/file/d/1-398iP4UtCQim7a1vOSHtBOnN4fqpcp/view?usp=drive_link
/ ├── audio/ │ ├── speaker_1/ │ │ ├── audio_1.wav │ │ ├── audio_2.wav │ │ └── ... │ ├──
speaker_2/ │ │ ├── audio_1.wav │ │ ├── audio_2.wav │ │ └── ... │ └── ... └── noise/ ├──
noise_1.wav ├── noise_2.wav └── ...
Where:
* `audio/` contains subfolders for each speaker, with each subfolder containing audio
recordings of that speaker.
* `noise/` contains audio recordings of background noise.
Before running the code, make sure to update the `DATASET_ROOT` variable in the
notebook to point to the location of your dataset.
## Training
To train the model, run the notebook `speaker_recognition.ipynb`. The notebook will load
the dataset, preprocess the audio data, build the model, and train it.
## Inference
To use the trained model for inference, you can use the following code:
Load the saved model
 model = tf.keras.models.load_model('model.keras')
Preprocess the audio input
def preprocess_audio(audio_path):
# Load the audio file
audio, sr = librosa.load(audio_ path, sr=16000)
# Convert the audio to a spectrogram
spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
# Convert the spectrogram to a log scale
log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
# Reshape the log spectrogram to the expected input shape of the model
input_data = log_spectrogram.reshape((1, log_spectrogram.shape[0], log_spectrogram.shap
e[1], 1))
return input_data
Make a prediction
audio_path = 'path/to/audio/file.wav' input_data = preprocess_audio(audio_path)
prediction = model.predict(input_data)
Get the predicted speaker label
predicted_label = np.argmax(prediction)
Print the predicted speaker label
 print('Predicted speaker:', predicted_label)
This code loads the saved model, preprocesses the audio input, and makes a prediction. The
`preprocess_audio` function handles the audio preprocessing steps, such as loading the aud
io file, converting it to a spectrogram, and reshaping it to the expected input shape of the m
odel. The `predict` method of the model is then called to make a prediction, and the `argma
x` function is used to get the predicted speaker label.
## Trained Models
The trained model file (`model.keras`) can be found in it
https://drive.google.com/file/d/1-398iP4UtCQim7a1vOSHtBOnN4fqpcp/view?usp=drive_link
Conclusion:
This speaker recognition system, implemented using a Convolutional Neural Network
(CNN) with residual blocks and trained on a noise-augmented dataset, demonstrates a high
accuracy of 0.9788 on the validation set. The system leverages the power of CNNs to
effectively extract relevant features from audio data, while the residual blocks enhance the
learning process and model performance. The use of noise augmentation during training
improves the system's robustness and generalization capability, enabling it to perform well
on unseen audio samples. These results suggest the potential of this system for real-world
speaker recognition applications, offering a promising solution for identifying speakers
based on their unique vocal characteristics.
