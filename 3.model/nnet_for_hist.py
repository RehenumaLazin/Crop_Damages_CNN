import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from datetime import datetime

print("Updated")

class Config:
    B, W, H, C = 32, 32, 20, 6  # For May-June H=20, for Jul-Nov H=50
    train_step = 10000
    lr = 1e-3
    weight_decay = 0.005
    keep_prob = 0.25
    load_path = r"/shared/stormcenter/rehenuma/ML_Project/AGU_2019_CropDamaged_Area_MetForcing/MN_MO/Corn/without_LW/Img_Output"
    save_path = r"/shared/stormcenter/rehenuma/ML_Project/Corn_Corrected/MN_MO_metForcing/May_June_without_LW_HAND/Extended_0.25_Dropout_10000_steps_32_bins_2017"

def conv2d(input_data, out_channels, filter_size, stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")

def conv_relu_batch(input_data, out_channels, filter_size, stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a, axes=[0, 1, 2])
        return tf.nn.relu(b)

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class NeuralModel:
    def __init__(self, config, name):
        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        conv1_1 = conv_relu_batch(self.x, 128, 3, 1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(conv1_1_d, 256, 3, 2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2_d, 256, 3, 1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1_d, 512, 3, 2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2_d, 512, 3, 1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)

        conv3_2 = conv_relu_batch(conv3_1_d, 1024, 3, 2, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)

        dim = np.prod(conv3_2_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_2_d, [-1, dim])

        self.fc6 = dense(flattened, 1024, name="fc6")
        self.fc6_2 = dense(self.fc6, 1024, name="fc6_2")
        self.logits = tf.squeeze(dense(self.fc6_2, 1, name="dense"))

        self.loss_err = tf.reduce_sum(tf.square(self.logits - self.y))
        self.loss_reg = tf.reduce_sum(tf.abs(self.logits - self.y))
        alpha = 5
        self.loss = self.loss_err + self.loss_reg * alpha

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('dense', reuse=True):
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')

        with tf.variable_scope('conv1_1/conv2d', reuse=True):
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')
