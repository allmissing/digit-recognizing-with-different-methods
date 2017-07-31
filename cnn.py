# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:00:43 2017

@author: 705family
"""
#%% 
import numpy as np
import pandas as pd
import tensorflow as tf
"import sklearn as sl"
from sklearn import preprocessing

#%% 数据预处理
print("loading dataset..........")
data = pd.read_csv("F:/my_paragrams/digit recognize/train.csv")
traindata = data.iloc[0:39999,:]
testdata = data.iloc[40000:41999,:]

#%% 
features = traindata.iloc[:,1:785]
labels = traindata['label']
test_features = testdata.iloc[:,1:785]
test_labels = testdata['label']

#%% 输出编码
features = (features.values).astype('float')
labels = (labels.values).astype('float')
test_features = (test_features.values).astype('float')
test_labels = (test_labels.values).astype('float')

enc = preprocessing.LabelBinarizer(sparse_output=False)
labels = enc.fit_transform(labels)
test_labels = enc.fit_transform(test_labels)


#%% 数据归一化处理
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
test_features = scaler.transform(test_features)
labels = scaler.fit_transform(labels)
test_labels = scaler.transform(test_labels)
"""
这里不能使用SKlearn的scale函数来处理，否则准确率只有0.09
"""

#%% 构建图
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
"truncated_normal产生正态分布的随机数"

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
"""
"节点"
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

"权值"
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import random
pin = np.linspace(0,40000-2,40000-2,dtype=int)
random.shuffle(pin)
for i in range(800):
    batch_x = features[pin[((i-1)*50):(i*50-1)],:]
    batch_y = labels[pin[((i-1)*50):(i*50-1)],:]
    train_step.run(feed_dict = {x:batch_x,y_:batch_y})
print(accuracy.eval(feed_dict = {x:test_features, y_:test_labels}))
"""

"""
网络
"""
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
"第一层"
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
"第二层"
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
"全连接层"
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
"dropout"
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
"输出层"
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)


"""
选择
"""
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

import random
pin = np.linspace(0,40000-2,40000-2,dtype=int)
random.shuffle(pin)
for i in range(800):
    batch_x = features[pin[((i-1)*50):(i*50-1)],:]
    batch_y = labels[pin[((i-1)*50):(i*50-1)],:]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict = {x:batch_x,y_:batch_y,keep_prob:0.5})


print("test accuracy %g"%accuracy.eval(feed_dict = {x:test_features, y_:test_labels,keep_prob:1.0}))

