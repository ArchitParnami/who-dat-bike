# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:26:42 2018

@author: aparnami
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard


np.random.seed(42)
tf.set_random_seed(100)
batch = 10
target_size = (256,256)
feature_detectors = 32
epochs=5
training_set_path ='bikes/training_set'
model_name='bike_classifier.h5'

def build_classifier():
    classifier = Sequential()
    classifier.add(Convolution2D(feature_detectors, (3,3), input_shape=(target_size[0], target_size[1], 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(feature_detectors, (3,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Convolution2D(feature_detectors, (3,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 512, activation = 'sigmoid'))
    classifier.add(Dense(units = 128, activation = 'sigmoid'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier

def get_training_set():
    train_datagen = ImageDataGenerator(rescale=1./255)
    training_set = train_datagen.flow_from_directory(training_set_path,
                                                     target_size=target_size,
                                                     batch_size=batch,
                                                     class_mode='binary')
    print("Class Labels -> ", training_set.class_indices)
    return training_set

def train_save(classifier, training_set):
    tbCallBack = TensorBoard(log_dir='./Graph',histogram_freq=0, write_graph=True, write_images=True)
    tbCallBack.set_model(classifier)
    classifier.fit_generator(training_set,epochs=epochs)
    print("Training finished. Saving the model as " + model_name)
    classifier.save(model_name)
    print("Model saved to disk")
    
if __name__ == '__main__':
    print("Building the model...")
    classifier = build_classifier()
    print("Reading the training set from path ./" + training_set_path)
    training_set = get_training_set()
    print("Starting Training..")
    train_save(classifier, training_set)
    