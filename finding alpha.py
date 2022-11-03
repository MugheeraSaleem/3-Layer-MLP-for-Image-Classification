# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:09:08 2022

@author: haier
"""

import matplotlib.pyplot as plt
import ANN as ann
import Functions as func
import numpy as np

Dataset_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset'
Prediction_files_path = 'E:/Masters/Semester 3/DEEP LEARNING/3 Layer ANN for Image classification/Horses_Duck_Dataset/Prediction Images'

x_train,y_train = ann.prep_train_data(Dataset_path)
x_test,y_test = ann.prep_test_data(Prediction_files_path)

##### changing shapes to make calculations possible ########

x_train = x_train.T
y_train = y_train.reshape(1,x_train.shape[1])

x_test = x_test.T
y_test = y_test.reshape(1,x_test.shape[1])

##################### printing shapes ######################

print('shape of x_train:',x_train.shape) # shape of x_train: ( 10000,20)  (n,m)
print('shape of y_train:',y_train.shape) # shape of y_train: (1,10000)


print('shape of x_test:',x_test.shape) # shape of x_test: ( 10000,10)  (n,m)
print('shape of y_test:',y_test.shape) # shape of y_test: (1,10000)

############### number of training examples and features #########

m_train = x_train.shape[1]
print('# of training examples is ',m_train)
n = x_train.shape[0]

ideal_list = [100,0]


def training_model (x, y, n_h1, n_h2, n_h3, learning_rate, iterations, lamda, epsilon):
    
    '''
    This function is used to train the model.
    
    The arguments are:
    
    x = featuere vector
    y = label vector
    n_h1 = hidden layer 1 nodes
    n_h2 = hidden layer 2 nodes
    learning rate = learning rate
    iterations = iterations
    lamda = regularization parameter
    epsilon = used for gradient checking
    
    This function returns:
    
    w1 = 1st layer weights vector
    w2 = 2nd layer weights vector
    w3 = 3rd layer weights vector
    b1 = 1 layer biases 
    b2 = 2nd layer biases
    b3 = 3rd layer biases
    
    '''
    
    
    n_x = x.shape[0]
    n_y = y.shape[0]
    
    #print(n_x,n_y)
    
    w1, b1, w2, b2, w3, b3, w4, b4 = func.init_thetas_biases(n_x, n_h1, n_h2, n_h3, n_y) # Function calling for intializing parameters
    
    cost_list = []    # empty list
    
   
    
    # for loop for iterations
    
    for i in range(iterations):
        
        z1, a1, z2, a2, z3, a3, z4, a4 = func.forward_propagation(x, w1, b1, w2, b2, w3, b3, w4, b4)  # function for forward propagation
        
        cost = func.cost_function(a4, y) + func.regularization(w1, w2, w3, w4, lamda, m_train) # cost + regularization
        
        dz4, dw4, db4, dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1 = func.back_propagation(x, y, w1, b1, w2,
                                                                                           b2, w3, b3, w4,
                                                                                           b4, z1, a1, z2, 
                                                                                           a2, z3, a3, z4, 
                                                                                           a4, lamda) # calling back propagation 
        
     
        
        w1, b1, w2, b2, w3, b3, w4, b4 = func.update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dz4, 
                                                        dw4, db4, dz3, dw3, db3,
                                                        dz2, dw2, db2, dz1, dw1, db1, 
                                                        learning_rate) # updating parameters 
        
        cost_list.append(cost)
        print(cost,' at ',i,'on learning_rate ',learning_rate)     # printing cost
    
    for cost in cost_list:
        if cost < ideal_list[0]:
            ideal_list[0]=cost
            ideal_list[1]=learning_rate
    
    return  w1, b1, w2, b2, w3, b3, w4, b4, cost_list,ideal_list    




n_h1 = 100  # number of neurons in hidden layer 1
n_h2 = 20  # number of neurons in hidden layer 2
n_h3 = 10   # number of neurons in hidden layer 3
iterations = 10  # number of iterations of model
lamda = 0.5       # regularization parameter
threshold = 0.5    
epsilon= 1e-7      # for gradient checking
cost_list = []
learning_rate_list = []

for learning_rate in np.arange(0.01,1,0.01):
    _, _, _, _, _, _, _, _, cost, ideal_list = training_model (x_train, y_train, n_h1, n_h2, n_h3, learning_rate, iterations, lamda, epsilon)
    learning_rate_list.append(learning_rate)    
    cost_list.append(cost)

plt.plot(learning_rate_list,cost_list) 
plt.show()
print('cost goes down best at learning rate ',ideal_list[1], 'cost is', ideal_list[0])