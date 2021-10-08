import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math
import json
from json import JSONEncoder
import os
import datetime
import time


dir_path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(dir_path, "train.csv")
test_path = os.path.join(dir_path, "test.csv")
data = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)
train_data = data[: (int((len(data))*0.8))]  # first 80 percent size 336000x784
# last 20 percent  size  8400 rows x 785
cv_data = data[(int((len(data))*0.8)):]

X_train = np.array(train_data.drop("label", axis=1).copy()
                   ).T  # take thelabel out
Y_train = np.array([train_data['label'].copy()])  # its column 1 of labels

X_cv = np.array(cv_data.drop("label", axis=1).copy()
                ).T  # its a row of length 785
Y_cv = np.array([cv_data['label'].copy()])  # its column 1 of labels

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_cv = np.array(X_cv)  # x cross validation
Y_cv = np.array(Y_cv)  # ycross validation

X_test = np.array(X_test).T
# # print(X_train.shape)  # size of (784, 33600)
# # print(X_test.shape)  # size of (784, 28000)
# # print(X_train.shape,Y_train.shape)#training data(784, 33600) (1, 33600)
# # print(X_cv.shape,Y_cv.shape)#testing data(784, 8400) (1, 8400)
# # print(Y_train.max())#maximum number is 9 from 0 to 9 MNIST data
# generating training data label converting into 10 x 33600
# Generating one_hot_vector so that later we can apply multi class classification
shape = (Y_train.max()+1, Y_train.shape[1])
# generatin a vector of size 10 rows and length 33600
one_hot = np.zeros(shape)
rows = np.arange(Y_train.shape[1])
one_hot[Y_train, rows] = 1  # replacing o with 1 in
Y_train = one_hot
# # print(shape)
# # print(Y_train)
# # print(one_hot.shape)
# # print(rows) #[    0     1     2 ... 33597 33598 33599]#
# # print(Y_cv.max())
# generating cross validation label data converting into 10 x 33600
# Generating one_hot_vector so that later we can apply multi class classification
shape = (Y_cv.max()+1, Y_cv.shape[1])
one_hot = np.zeros(shape)  # generating a hotvector of size 10x33600
rows = np.arange(Y_cv.shape[1])
one_hot[Y_cv, rows] = 1
Y_cv = one_hot
# # print(Y_cv)
# # print(rows)#[   0    1    2 ... 8397 8398 8399]

# # print(Y_train.shape,Y_cv.shape)#20 % data have been made an arrray of (10, 33600) (10, 8400)


X_train = np.array(X_train/255)  # normalizing data(784, 33600) (1, 33600)
X_cv = np.array(X_cv/255)  # normalizing 20 percent  size  8400 rows x 785

X_test = np.array(X_test/255)  # normlaizing test data (784, 28000)
# # print(X_test.shape)

n_X = X_train.shape[0]  # the value for n_X=784#
m = X_train.shape[1]  # the value for n_X=33600#

# Softmax gives a good classification as it standardizes the output values.
# def softmax(A):
# Softmax function outputs a vector that represents the probability distributions of a list of potential outcomes.
#     A = A - np.max(A)
#     return np.exp(A) / np.sum(np.exp(A), axis=0)


nums = np.array([4, 5, 6])
# # print(softi.softmax(nums))
# # print(n_X)
# # print(m)


m = X_train.shape[1]
mini_batches = []

permutation = list(np.random.permutation(m))
shuffled_X = X_train[:, permutation]
shuffled_Y = Y_train[:, permutation]
mini_batch_N = []


def random_mini_batches(X, Y, mini_batch_size=32):  # the value for n_X=33600#
    m = X.shape[1]
    mini_batches = []
    print(m)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    num_comp_batchs = math.floor(m/mini_batch_size)
    print(num_comp_batchs)
    for i in range(num_comp_batchs):
        mini_batch_X = shuffled_X[:, i*mini_batch_size: (i+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i*mini_batch_size: (i+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            end = m - mini_batch_size*math.floor(m/mini_batch_size)
            mini_batch_X = shuffled_X[:, num_comp_batchs*mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_comp_batchs*mini_batch_size:]
            # size of mini_batch_X is (784x64)
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    return mini_batches


mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size=32)
# n=0
for n in range(0, 1):
    for bt in range(1050*n, (n+1)*1050):
        mini_batch_N.append(mini_batches[bt])
layer_dims = [784, 1000, 500, 250, 100, 50, 10]
L = len(layer_dims)git
parameters1 = {}
for l in range(1, L):
    parameters1["W"+str(l)] = np.random.randn(layer_dims[l],
                                              layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])

    parameters1["b"+str(l)] = np.zeros((layer_dims[l], 1))*0.01


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


with open(os.path.join(dir_path, "TrainData1.json"), "w+") as write_file:
    json.dump(mini_batch_N[0:1050], write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")

with open(os.path.join(dir_path, "Parameters1.json"), "w+") as write_file:
    json.dump(parameters1, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")


# with open(os.path.join(dir_path,"TrainData2.json"), "w+") as write_file:
#json.dump(mini_batch_N[525:1050], write_file, cls=NumpyArrayEncoder)
#print("Done writing serialized NumPy array into file")

# with open(os.path.join(dir_path,"TrainData3.json"), "w+") as write_file:
#    json.dump(mini_batch_N[700:1050], write_file, cls=NumpyArrayEncoder)
#print("Done writing serialized NumPy array into file")


# with open(os.path.join(dir_path,"TrainData4.json"), "w+") as write_file:
#     json.dump(mini_batch_N[786:1050], write_file, cls=NumpyArrayEncoder)
#print("Done writing serialized NumPy array into file")

# # for deserialization
#print("Decode JSON serialized numpy array")
# with open(os.path.join(dir_path,"TrainData.json"), "r+") as read_file:
#   print("deserializing")
#  decodeArray=json.load(read_file)
# finaldat=np.asarray(decodeArray)
# print(finaldat)
