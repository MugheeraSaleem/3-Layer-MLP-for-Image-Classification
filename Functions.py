# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:22:21 2022

@author: haier
"""


import numpy as np
from math import sqrt


def relu(x,derivative = False):
    '''
    First, second and Third hidden layer activation function and it's derivative.'
    '''
    a = np.array(x > 0, dtype = np.float32) if (derivative) else np.maximum(x, 0)
    return a
     
def tanh(x,derivative = False):
    '''
    Second hidden layer activation function and it's derivative.
    '''
    a = (1 -  np.power(np.tanh(x), 2)) if (derivative) else np.tanh(x)
    return a 

def sigmoid(x):
    '''
    Output layer activation function.
    '''
    return 1/(1 + np.exp(-x))

def xavier_uniform_init(fan_in,fan_out,size):
    '''
    initialization for sigmoid.
    '''
    lower = -sqrt(6/fan_in+fan_out) 
    upper = sqrt(6/fan_in+fan_out)
    return np.random.uniform(low=lower, high=upper, size=size)

def he_uniform_init(fan_in,fan_out,size):
    ''''
    Initialization for relu.
    '''
    lower = -sqrt(6/fan_in) 
    upper = sqrt(6/fan_in)
    return np.random.uniform(low=lower, high=upper, size=size)
    
def init_thetas_biases(n_x , n_h1 , n_h2 , n_h3, n_y):  # (40000,50,25,10,1)  (input,h1,h2,h3,output)
    '''
    n_x = neurons in input layer
    n_h1 = neurons in the 1st hidden layer
    n_h2 = neurons in the 2nd hidden layer
    n_h3 = neurons in the 3rd hidden layer
    n_y = neurons in the output layer
    
    
    ########## HE_normal_init #########
    W_1 = np.random.randn(n_1,n)*np.sqrt(2.0/(n))
    W_2 = np.random.randn(n_2,n_1)*np.sqrt(2.0/(n_1+1))
    W_3 = np.random.randn(n_3,n_2)*np.sqrt(2.0/(n_2+1))
    
    '''
    ###### weights and bias for input to hidden layer 1 ###########
    
    np.random.seed(7)
    
    #w1 = np.random.randn(n_h1,n_x)*0.001  #     (100,10000)
    #b1 = np.zeros((n_h1,1))              #     (100,1)
    
    #w1 = he_uniform_init(fan_in = n_x,fan_out = n_h1,size = (n_h1,n_x)) * 0.001
    w1 = np.random.randn(n_h1,n_x)*np.sqrt(2.0/(n_x))
    b1 = np.random.uniform(low= -sqrt(6/n_x) , high= sqrt(6/n_x) , size=(n_h1,1))
    
    ###### weights and bias for hidden layer 1 to hidden layer 2 ###########
    
    #w2 = np.random.randn(n_h2,n_h1)*0.001 #    (50,100)
    #b2 = np.zeros((n_h2,1))               #    (50,1)
    
    #w2 = he_uniform_init(fan_in = n_h1,fan_out = n_h2,size = (n_h2,n_h1)) * 0.001
    w2 = np.random.randn(n_h2,n_h1)*np.sqrt(2.0/(n_h1))
    b2 = np.random.uniform(low= -sqrt(6/n_h1) , high= sqrt(6/n_h1) , size=(n_h2,1))
    
    ###### weights and bias for hidden layer 2 to hidden layer 3 ###########
    
    #w3 = np.random.randn(n_h3,n_h2)*0.001 #    (25,50)
    #b3 = np.zeros((n_h3,1))               #    (25,1)
    
    #w3 = he_uniform_init(fan_in = n_h2,fan_out = n_h3,size = (n_h3,n_h2)) * 0.001
    w3 = np.random.randn(n_h3,n_h2)*np.sqrt(2.0/(n_h2))
    b3 = np.random.uniform(low= -sqrt(6/n_h2) , high= sqrt(6/n_h2) , size=(n_h3,1))
    
    ###### weights and bias for hidden layer 3 to output ###########
    
    #w4 = np.random.randn(n_y,n_h3)*0.001 #     (1,25)
    #b4 = np.zeros((n_y,1))               #     (1,1)
    
    #w4 = xavier_uniform_init(fan_in = n_h3,fan_out = n_y,size = (n_y,n_h3)) * 0.001
    w4 = np.random.randn(n_y,n_h3)*np.sqrt(2.0/(n_h3)) 
    b4 = np.random.uniform(low= -sqrt(6/n_h3+n_y) , high= sqrt(6/n_h3+n_y) , size=(n_y,1))
    
    return w1, b1, w2, b2, w3, b3, w4, b4     # parameters

def forward_propagation(x, w1, b1, w2, b2, w3, b3, w4, b4):
    '''
    Forward propagation and prediction on specific weights
    '''
    
    ############ hiden Layer 1 ########
    
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)
    
    ############ hiden Layer 2 #########
    
    z2 = np.dot(w2,a1) + b2
    a2 = relu(z2)
    
    ############ hiden Layer 3 #########
    
    z3 = np.dot(w3,a2) + b3
    a3 = tanh(z3)
    
    ###### output layer ######
    
    z4 = np.dot(w4,a3) + b4
    a4 = sigmoid(z4)
    
    return z1, a1, z2, a2, z3, a3, z4, a4 # forward_cache

def regularization(w1, w2, w3, w4, lamda, m):
    
    '''
    this function is used for regularization calculation.
    '''
    reg = (lamda/(2*(m))*(np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)) + np.sum(np.square(w4)) ))
    #print(reg)
    return reg

def cost_function(a4,y):
    '''
    Finding cost of a certain model
    
    inputs:
        a4 = prediction
        y = label vector
    
    outputs:
        cost
    '''
    m = y.shape[1]     # Number of examples 
    y_1 = np.log10(a4+0.0000000001)       # 0.0000000001 added to avoid overflow as log of 0 is infinity
    y_0 = np.log10(1-a4+0.000000000001)
    cost = ((-1/m)*(np.sum((y * y_1) + ((1-y)*y_0))))
    return cost

def back_propagation(x, y, w1, b1, w2, b2, w3, b3, w4, b4, z1, a1, z2, a2, z3, a3, z4, a4, lamda):
    
    '''
    This function is used for calculating the gradient at any time in order to update the parameters.
    
    inputs:
        x, feature vector
        y, label vector
        w1,layer 1 weights
        b1, layer 1 biases
        w2, layer 2 weights
        b2, layer 2 biases
        w3, layer 3 weights
        b3, layer 3 biases
        w4, Output layer weights
        b4, Output layer biases
        z1, layer 1 calculation
        a1, layer 1 activation
        z2, layer 2 calculation
        a2, layer 2 activation
        z3, layer 3 calculation
        a3, layer 3 activation
        z4, output layer calculation
        a4, output layer activation
        lamda regularization parameter
    
    outputs:
        dz4, dw4, dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1
        all gradients 
    
    # FOR REFERENCE
    # dz3 = a3-y
    # dw3 = (1/ m) * dz3.dot(a2.T) #+ ((lamda/m)*w3)
    # db3 = (1/ m) * np.sum(dz3)
    
    # dz2 = w3.T.dot(dz3) * d_tanh(z2)
    # dw2 = (1/ m) * dz2.dot(a1.T) #+ ((lamda/m)*w2)
    # db2 = (1/ m) * np.sum(dz2)
    
    # dz1 = w2.T.dot(dz2) * d_relu(z1)
    # dw1 = (1/ m) * dz1.dot(x.T) #+ ((lamda/m)*w1)
    # db1 = (1/ m) * np.sum(dz1)
    
    '''
    m = x.shape[0]
    
    # Output Layer
    
    dz4 = a4-y
    dw4 = (1/ m) * dz4.dot(a3.T) #+ ((lamda/m)*w4)
    db4 = (1/ m) * np.sum(dz4)
    
    # 3rd Hidden Layer
    
    dz3 = w4.T.dot(dz4) * tanh(z3,derivative = True)
    dw3 = (1/ m) * dz3.dot(a2.T) #+ ((lamda/m)*w3)
    db3 = (1/ m) * np.sum(dz3)
    
    # 2nd Hidden Layer
    
    dz2 = w3.T.dot(dz3) * relu(z2,derivative = True)
    dw2 = (1/ m) * dz2.dot(a1.T) #+ ((lamda/m)*w2)
    db2 = (1/ m) * np.sum(dz2)
    
    # 1st Hidden Layer
    
    dz1 = w2.T.dot(dz2) * relu(z1,derivative = True)
    dw1 = (1/ m) * dz1.dot(x.T) #+ ((lamda/m)*w1)
    db1 = (1/ m) * np.sum(dz1)
    
    return dz4, dw4, db4, dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1 # gradients

def update_parameters(w1, b1, w2, b2, w3, b3, w4, b4, dz4, dw4, db4, dz3, dw3, db3,
                      dz2, dw2, db2, dz1, dw1, db1, 
                       learning_rate):
    '''
    This function is updating the parameters based on thier gradients.
    
    Inputs:
        All the weights and biases and the gradients are inputs in the form, w1, b1, w2, b2, w3, b3, dz3, dw3, db3
                              , dz2, dw2, db2, dz1, dw1, db1,
        learning_rate: learning rate
    
    outputs:
        w1, b1, w2, b2, w3, b3 
        updated weights and biases
    '''
    
    w1 = w1-learning_rate*dw1
    #print('shape of w1 is',w1.shape)
    b1 = b1-learning_rate*db1
    
    w2 = w2-learning_rate*dw2
    #print('shape of w2 is',w2.shape)
    b2 = b2-learning_rate*db2
    
    w3 = w3-learning_rate*dw3
    #print('shape of w3 is',w3.shape)
    b3 = b3-learning_rate*db3
    
    w4 = w4-learning_rate*dw4    
    #print('shape of w4 is',w4.shape)
    b4 = b4-learning_rate*db4

    
    return w1, b1, w2, b2, w3, b3, w4, b4   # parameters


def errors(x,y,parameters,threshold):
    
    '''
    This function calculates all the metrics for the model evaluation on a provided dataset.
    
    Arguments:
        x = feature vector
        y = label vector
        parameters = parameters
        threshold = threshold for counting tp,tn,fp,fn
    outputs:
        Accuracy,
        precision,
        recall,
        f1_score,
        tp = True positives
        tn = True negatives
        fp = false positives
        fn = false negatives
    '''
    
    m = x.shape[1]            # Number of examples in the set
    
    ########## opening dictionary to get parameters ##########
    
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    w3=parameters['w3']
    b3=parameters['b3']
    w4=parameters['w4']
    b4=parameters['b4']
    
    tp = 0      # True positives
    tn = 0      # True negatives
    fp = 0      # false positives
    fn = 0      # false negatives

    _, _, _, _, _, _, _, a4 = forward_propagation(x, w1, b1, w2, b2, w3, b3, w4, b4)  # calling forward propagation function
    
    for h in range(0,m,1):                       # loop iterated over all the examples
        if a4[0][h] >= threshold and y[0][h] == 1:    # counting true positives
            tp=tp+1
        elif a4[0][h] < threshold and y[0][h] == 0:   # counting true negatives
            tn=tn+1
        elif a4[0][h] >= threshold and y[0][h] == 0:  # counting false positives
            fp=fp+1
        elif a4[0][h] < threshold and y[0][h] == 1:   # counting false negatives
            fn=fn+1
            
    Accuracy = (tp+tn)*(100/m)
    
    precision = tp/(tp+fp)
    
    #print('Precision is: ',"{0:.2f}".format(precision))
    
    recall = tp/(tp+fn)
    
    #print('Recall is: ',"{0:.2f}".format(recall))
    
    f1_score = 2*((precision*recall)/(precision+recall))
    
    #print('f1_score is ',"{0:.2f}".format(f1_score))
    
    return Accuracy,precision,recall,f1_score,tp,tn,fp,fn

def testing(x,y,parameters):
    '''
    This function returns the cost of a model on a dataset.
    
    inputs:
        x : feature vector
        y : label vector
        parameters: trained model
    
    outputs:
        cost
    '''
    
    ########## Unpacking the dictionary to get the parameters #######

    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']
    w3=parameters['w3']
    b3=parameters['b3']
    w4=parameters['w4']
    b4=parameters['b4']    
    ########### hidden layer 1 ##########
    
    z1 = np.dot(w1,x) + b1
    a1 = relu(z1)
    
    ############ hiden Layer 2 #########
    
    z2 = np.dot(w2,a1) + b2
    a2 = relu(z2)
    
    ############ hiden Layer 3 #########
    
    z3 = np.dot(w3,a2) + b3
    a3 = tanh(z3)
    
    ###### output layer ######
    
    z4 = np.dot(w4,a3) + b4
    a4 = sigmoid(z4)
    
    m = y.shape[1]     # Number of examples 
    y_1 = np.log10(a4+0.0000000001)
    y_0 = np.log10(1-a4+0.000000000001)
    cost = ((-1/m)*(np.sum((y * y_1) + ((1-y)*y_0))))
    
    return cost