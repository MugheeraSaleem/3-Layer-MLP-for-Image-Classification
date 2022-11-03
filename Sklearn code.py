# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:23:50 2022

@author: haier
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import ANN as ann



Dataset_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset'
Prediction_files_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset/Prediction Images'

x_train,y_train = ann.prep_train_data(Dataset_path)
x_test,y_test = ann.prep_test_data(Prediction_files_path)

classifier = MLPClassifier(hidden_layer_sizes=(5,5,5), max_iter=1000,activation = 'relu',solver='adam',random_state=1,alpha=0.00095)
classifier.fit(x_train, y_train)
y_pred_train = classifier.predict(x_train)
y_pred_test = classifier.predict(x_test)
print('accuracy for training set is: '+ str(accuracy_score(y_train,y_pred_train)))
print('accuracy for test set is: '+ str(accuracy_score(y_test,y_pred_test)))