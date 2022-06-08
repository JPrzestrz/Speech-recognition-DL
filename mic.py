import wave
import pyaudio
# Libraries for voice and deep
import tensorflow as tf
import numpy as np
import pathlib
import os
from func import decode_audio, get_label, get_waveform_and_label, get_spectrogram, plot_spectrogram, get_spectrogram_and_label_id, preprocess_dataset

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "go/output.wav"
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("* recording")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

model = tf.keras.models.load_model('saved_model/my_model')

# CHECKING VOICE
DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
AUTOTUNE = tf.data.AUTOTUNE

MY_TEST_PATH = 'go/'
data_dir = pathlib.Path(MY_TEST_PATH)
# Creating tf of all of my files from mydata dir 
# !IMPORTANT! Very important note is to check 
# whether the wav file is mono and not stereo 
# and it is 16k MHz and it's good if it's normalized ! ! !
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
#print('\n\nNumber of total examples:', len(filenames))
# Processing Files
test_ds = preprocess_dataset(filenames)

# Dividing into audio and labels 
# no or wrong labels in mydata files
test_audio = []
test_labels = []
for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())
test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

# Using our trained model to predict audio files
# included in mydata dir
y_pred = np.argmax(model.predict(test_audio), axis=1)
print(y_pred)
# Txt labels printing **testing purposes** 
for i in y_pred:
      print(commands[i])

model.save('saved_model/my_model')