# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:00:43 2017

@author: 705family
"""
#%% 
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn as sl

#%% 数据预处理
print("loading dataset..........")
data = pd.read_csv("F:/my_paragrams/digit recognize/train.csv")
traindata = data.iloc[0:20999,:]
testdata = data.iloc[21000:41999,:]

features = traindata.iloc[:,1:785]
labels = traindata['label']
test_features = testdata.iloc[:,1:785]
test_labels = testdata['label']

features = (features.values).astype('float')
labels = (labels.values).astype('float')
test_features = (test_features.values).astype('float')
test_labels = (test_labels.values).astype('float')

enc = sl.preprocessing.OneHotEncoder(sparse=False)
labels = enc.fit_transform(labels)
test_labels = enc.fit_trainsform(test_labels)

#%% 构建图

"节点"
x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])

"权值"
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_tables())

y = tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

import random
pin = np.linspace(0,21000-1,21000-1,dtype=int)
random.shuffle(pin)

for i in range(420):
    batch_x = features[pin[((i-1)*50):(i*50-1)],:]
    batch_y = labels[pin[((i-1)*50):(i*50-1)]]
    train_step.run(feed_dict = {x:batch_x,y_:batch_y})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(accuracy.eval(feed_dict = {x:test_features, y_:test_labels}))
