import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math


def DNN_model(data, parameters, layer_dims, num_iterations, learning_rate):
#     mini_batches = rando.random_mini_batches(X, Y, mini_batch_size=32)
    layer_dims = [784, 1000, 500, 250, 100, 50, 10]
#     dc_val=[0.050507627227610534,0.044721359549995794,0.06324555320336758,0.08944271909999159,0.1414213562373095,0.2]
    parameters = {}
    L = len(layer_dims)
    m = len(data)*32
    print(type(data))
#     Y_prediction = np.zeros((1, m))
    forward_cache = {}
    L = len(layer_dims)
    grads = {}
    costs = []
    mini_batch_N=[]
    mini_batch_X=[]
    mini_batch_Y=[]
    cost_1 = 1000
    cost_2 = 1000
    
    for l in range(1, L):
         parameters["W"+str(l)] = np.random.randn(layer_dims[l],
                                             layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
     
         parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))*0.01
    def softmax(A):
    # Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes.
        A = A - np.max(A)
        return np.exp(A) / np.sum(np.exp(A), axis=0)
    
    for i in range(0, num_iterations):
        for mini_batch in data:
             (mini_batch_X, mini_batch_Y) = mini_batch
             mini_batch_X=np.array(mini_batch_X) 
             mini_batch_Y=np.array(mini_batch_Y)
             forward_cache["A0"] = np.copy(mini_batch_X)
             A = np.copy(mini_batch_X)
             
             for l in range(1, L-1):
                 A_prev = np.copy(A)
        # forward chache refer to the training of the model
                 forward_cache["Z"+str(l)] = np.dot(parameters["W"+str(l)],
                                                   A_prev)+parameters["b"+str(l)]  # weights*data+bias
                 A = np.maximum(0, forward_cache["Z"+str(l)])

                 forward_cache["A"+str(l)] = np.copy(A)
        # last_layer
             W = parameters["W"+str(L-1)]  # weight taken from the last layer
             b = parameters["b"+str(L-1)]  # bias taken fromthe ;ast layer
             forward_cache["Z"+str(L-1)] = np.dot(W, A)+b
             AL_train = softmax(forward_cache["Z"+str(L-1)])
        # Softmax gives a good classification as it standardizes the output values.
        #        print(AL_train.shape)
        # compute_cost
             m = mini_batch_Y.shape[1]
        #  print(mini_batch_Y.shape,AL_train.shape)
             cost_1 = -np.mean(mini_batch_Y * np.log(AL_train + 1e-8))
        # print(np.sum(W))
        # print(np.sum(b))
             print("After {} epoch cost is {}".format(i, cost_1))
             costs.append(float(cost_1))
        # backward_prop
             grads["dZ"+str(L-1)] = AL_train-mini_batch_Y
             grads["dW"+str(L-1)] = (1/m)*(np.dot(grads["dZ" +
                                                       str(L-1)], forward_cache["A"+str(L-2)].T))
             grads["db"+str(L-1)] = (1/m) * \
                 (np.sum(grads["dZ"+str(L-1)], axis=1, keepdims=True))
       
             for l in reversed(range(1, L-1)):
                 grads["dZ"+str(l)] = np.multiply(np.dot(parameters["W"+str(l+1)].T,
                                                        grads["dZ"+str(l+1)]), np.int64(forward_cache["A"+str(l)] > 0))
                 grads["dW"+str(l)] = (1/m)*(np.dot(grads["dZ" +
                                                         str(l)], (forward_cache["A"+str(l-1)]).T))
                 grads["db"+str(l)] = (1/m) * \
                    (np.sum(grads["dZ"+str(l)], axis=1, keepdims=True))
             for l in range(1, L):
                 parameters["W" + str(l)] = parameters["W" + str(l)] -\
                     learning_rate * grads["dW" + str(l)]
                 parameters["b" + str(l)] = parameters["b" + str(l)] -\
                     learning_rate * grads["db" + str(l)]
        
    return parameters, costs, AL_train
  
