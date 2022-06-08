# TESTING FILE 
import tensorflow as tf
import pathlib
from func import preprocess_dataset, commands
import numpy as np
model = tf.keras.models.load_model('saved_model/my_model')

# Predicting my own recordings to check how it would work
'''
      Getting wav files from mydata directory and making predictions 
'''
# Getting my file from mydata dir 
MY_TEST_PATH = 'mydata/left'
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
# Txt labels printing **testing purposes** 
for i in y_pred:
      print(commands[i])

model.save('saved_model/my_model')