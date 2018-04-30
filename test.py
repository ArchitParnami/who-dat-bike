# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:02:45 2018

@author: aparnami
"""

import numpy as np
import os
from keras.preprocessing import image
from keras.models import load_model
import cv2

#Reusing this function from here
#https://github.com/MicrocontrollersAndMore/TensorFlow_Tut_2_Classification_Walk-through/blob/master/test.py
def writeResultOnImage(openCVImage, resultText):
    SCALAR_BLUE = (255.0, 0.0, 0.0)
    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1.0
    fontThickness = 2
    fontThickness = int(fontThickness)
    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)
    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)

target_size = (256,256)
model_name='bike_classifier.h5'
test_set_path ='bikes/test_set'

print("Trying to load the model " + model_name)
classifier = load_model(model_name)
print("Model loaded")
print("Test set path is ./" + test_set_path)
print()

for folder in os.listdir(test_set_path):
    folder_path = os.path.join(test_set_path, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        print("Testing File :\t", file_path)
        
        test_image = image.load_img(file_path, target_size=target_size)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        
        pred_class = classifier.predict_classes(test_image)
        prob = classifier.predict_proba(test_image)
        pred_class = pred_class[0][0]
        prob = prob[0][0]
        
        result_text = ""
        if pred_class == 0:
            score = round((1-prob)*100,3)
            result_text = 'Mountain Bike, {}% Confidence'.format(score)
        else:
            score = round(prob*100,3)
            result_text = 'Road Bike, {}% Confidence'.format(score)
        
        print("Prediction   :\t", result_text)
        print()
        
        img = cv2.imread(file_path)
        writeResultOnImage(img, result_text)
        cv2.imshow(file, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       

