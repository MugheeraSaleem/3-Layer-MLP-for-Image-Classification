# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:31:04 2022

@author: haier
"""

import numpy as np
import cv2
import os.path


Dataset_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset'
Prediction_files_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset/Prediction Images'
labels = ["Duck","Horse"]




def pre_processing(image_path):
    '''
    Images are first converted from RGB to grey and then the images are reduced in size. Then these reducded images
    are normalized by dividing them with 255. 
    
    Input:
    - image_path: This is the path of the image that is to be pre_processed.
    
    Returns:
    - img_pred: The processed image, it is first converted into grey and then resized to 100x100 and then normalized.
    
    '''
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_pred = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    
    img_pred = np.asarray(img_pred)
    
    img_pred = img_pred / 255
    return img_pred



def prep_train_data(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_train = np.array(X)   # Shape is (20,10000) (r,c) 
    
    y_duck =  np.zeros((10,1))
    y_horse = np.ones((10,1))
    Y_train = np.concatenate((y_duck,y_horse))     # Shape is (20,1) (r,c)
    
    ####### ZERO CONTERING ########
    
    X_train = X_train - np.mean(X_train,axis=0)
    X_train = X_train / np.std(X_train,axis=0)
    return X_train,Y_train

def prep_test_data(path):
    X = []
    for file in os.listdir(path):
            if (os.path.isfile(path + "/" + file)):
                image = pre_processing(path + "/" + file)
                image = np.reshape(image,(image.shape[0]*image.shape[1]))
                X.append(image)
    
    X_test = np.array(X)   # Shape is (10,10000) (r,c) 
    
    y_duck =  np.zeros((5,1))
    y_horse = np.ones((5,1))
    Y_test = np.concatenate((y_duck,y_horse))     # Shape is (10,1) (r,c)
    
    ####### ZERO CONTERING ########
    
    X_test = X_test - np.mean(X_test,axis=0)
    X_test = X_test / np.std(X_test,axis=0)

    return X_test,Y_test


# X_train,Y_train = prep_train_data(Dataset_path)
# print(X_train.shape)
# print(Y_train.shape)



            
            
            
            