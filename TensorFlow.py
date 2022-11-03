# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:41:17 2022

@author: haier
"""

import tensorflow as tf
import cv2
import numpy as np

Dataset_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/TENSORFLOW_DATA'


ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory=Dataset_path,
    labels='inferred',
    label_mode='binary',
    class_names=['Duck','Horse'],
    color_mode='grayscale',
    batch_size=2,
    image_size=(100, 100),
    validation_split=0.1,
    seed=1,
    subset='training',
) 

ds_val = tf.keras.preprocessing.image_dataset_from_directory(
    directory=Dataset_path,
    labels='inferred',
    label_mode='binary',
    class_names=['Duck','Horse'],
    color_mode='grayscale',
    batch_size=2,
    image_size=(100, 100),
    validation_split=0.1,
    seed=1,
    subset='validation',
) 


model = tf.keras.models.Sequential([
    tf.keras.layers.Input((100,100,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(ds_train,epochs=10,validation_data=ds_val,verbose=2)


prediction_path='E:/Masters/Semester 3/DEEP LEARNING/prediction.jpg'
img = cv2.imread(prediction_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_pred = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
img_pred = tf.keras.preprocessing.image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

print(img_pred.shape)

print(model.predict(img_pred))