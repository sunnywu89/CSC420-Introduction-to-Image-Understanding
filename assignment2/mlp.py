######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################

import numpy as np

def cross_entropy_loss_function(prediction, label):
    #TODO: compute the cross entropy loss function between the prediction and ground truth label.
    # prediction: the output of a neural network after softmax. It can be an Nxd matrix, where N is the number of samples,
    #           and d is the number of different categories
    # label: The ground truth labels, it can be a vector with length N, and each element in this vector stores the ground truth category for each sample.
    # Note: we take the average among N different samples to get the final loss.
    targets = np.eye(prediction.shape[1])[label]
    loss = np.sum(-targets *np.log(prediction)) / label.shape[0]
    return loss

def sigmoid(x):
    # TODO: compute the softmax with the input x: y = 1 / (1 + exp(-x))
    return 1 / (1 + np.exp(-x)) 

def softmax(x):
    # TODO: compute the softmax function with input x.
    #  Suppose x is Nxd matrix, and we do softmax across the last dimention of it.
    #  For each row of this matrix, we compute x_{j, i} = exp(x_{j, i}) / \sum_{k=1}^d exp(x_{j, k})
    exps = np.exp(x)
    sum = np.sum(exps, axis=1) # row
    sum = sum.reshape(sum.shape[0], -1)
    softmax = exps / sum
    return softmax

class OneLayerNN():
    def __init__(self, num_input_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a one-layer MLP.
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.W = np.random.randn(num_output_unit, num_input_unit)
        self.b = np.zeros((num_output_unit, 1))

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is an Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute output: z = softmax (input_x * W_1 + b_1), where W_1, b_1 are weights, biases for this layer
        # Note: If we only have one layer in the whole model and we want to use it to do classification,
        #       then we directly apply softmax **without** using sigmoid (or relu) activation
        self.z = (np.dot(self.W, input_x.T)+self.b.reshape(self.b.shape[0], -1)).T
        self.y = softmax(self.z)
        return self.y


    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        
        #compute gradient
        self.ts = np.eye(self.b.shape[0])[label]
        self.dz = (self.y - self.ts) / input_x.shape[0]
        self.dw = np.dot(self.dz.T, input_x)
        self.db = np.dot(self.dz.T, np.ones(input_x.shape[0]))
        # update weight and bias
        self.W = self.W - learning_rate*self.dw
        self.b = self.b = learning_rate*self.db


# [Bonus points] This is not necessary for this assignment
class TwoLayerNN():
    def __init__(self, num_input_unit, num_hidden_unit, num_output_unit):
        #TODO: Random Initliaize the weight matrixs for a two-layer MLP with sigmoid activation,
        # the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix: zero mean and the variance equals to 1
        #       and initialize the bias matrix as full zero using np.zeros()
        self.W1 = np.random.randn(num_hidden_unit, num_input_unit) 
        self.b1 = np.zeros((num_hidden_unit,1)) 
        self.W2 = np.random.randn(num_output_unit, num_hidden_unit)
        self.b2 = np.zeros((num_output_unit,1)) 

    def forward(self, input_x):
        #TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is Nxd matrix, where N is the number of samples and d is the number of dimension for each sample.
        # Compute: first layer: z = sigmoid (input_x * W_1 + b_1) # W_1, b_1 are weights, biases for the first layer
        # Compute: second layer: o = softmax (z * W_2 + b_2) # W_2, b_2 are weights, biases for the second layer
        self.z1 = (np.dot(self.W1, input_x.T) +self.b1).T # 64x50
        self.h = sigmoid(self.z1) 
        self.z2 = (np.dot(self.W2, self.h.T) + self.b2).T # 64x10
        self.y = softmax(self.z2) 
        return self.y


    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        #TODO: given the computed loss (a scalar value), compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass, such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        
        # compute gradient
        self.ts = np.eye(self.b2.shape[0])[label]
        self.dz2 = (self.y - self.ts) / input_x.shape[0] # 64x10
        self.dw2 = np.dot(self.dz2.T, self.h) # 10x50
        self.db2 = np.dot(self.dz2.T, np.ones(input_x.shape[0])) # 10x1
        self.dh = np.dot(self.dz2, self.W2) # 64x50
        # f(x) = sigmoid(x), f'(x) = f(x)*(1-f(x))
        self.dz1 = self.dh*sigmoid(self.z1)*(1-sigmoid(self.z1)) # 64x50
        self.dw1 = np.dot(self.dz1.T, input_x) # 64x10
        self.db1 = np.dot(self.dz1.T, np.ones(input_x.shape[0]))
        # update weight and bias
        self.W1 = self.W1 - learning_rate * self.dw1
        self.b1 = self.b1 - learning_rate * self.db1.reshape(self.db1.shape[0], -1)
        self.W2 = self.W2 - learning_rate * self.dw2
        self.b2 = self.b2 - learning_rate * self.db2.reshape(self.db2.shape[0], -1)
        
