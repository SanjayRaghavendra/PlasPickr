import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pathlib

#load dataset
path='C:/Users/Admin/.keras/datasets/data_final'
data_dir = tf.keras.utils.get_file('data_final', origin=path, untar=False)
data_dir = pathlib.Path(data_dir)

img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
#print(class_names)

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=7,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(7, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

epochs=10
history = resnet_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

import cv2
image=cv2.imread('C:/Users/Admin/Desktop/waste2.jpg')
image_resized=cv2.resize(image, (img_height,img_width))
image=np.expand_dims(image_resized,axis=0)

pred=resnet_model.predict(image)
print(pred)

output_class=class_names[np.argmax(pred)]
print(class_names)
print("The predicted class is", output_class)

path="C:/Users/Admin/Desktop/Preliminary_model"
resnet_model.save(path)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(resnet_model)
tflite_model = converter.convert()

# Save the model.
with open('model1.tflite', 'wb') as f:
  f.write(tflite_model)