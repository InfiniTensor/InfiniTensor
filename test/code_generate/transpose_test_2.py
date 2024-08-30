import tensorflow as tf
import numpy as np

def h_d2_d1(i0):
   # i0 = tf.reshape([i for i in range(8*8)], [1,1,8,8])
    i1 = tf.reshape(i0, [1,1,4,2,8])
    i2 = tf.transpose(i1, [0,1,3,2,4])
    i3 = tf.reshape(i2, [1,1,8,8])
    return i3

def h_d1_d2(i0):
   # i0 = h_d2_d1(tf.reshape([i for i in range(8*8)], [1,1,8,8]))
   i1 = tf.reshape(i0, [1,1,2,4,8])
   i2 = tf.transpose(i1, [0,1,3,2,4])
   i3 = tf.reshape(i2, [1,1,8,8])
   return i3


def nd2_hm2(i0):
    # i0 = tf.reshape([i for i in range(2*4*4)], [2,1,4,4])
    i1 = tf.reshape(i0, [1,2,1,4,4])
    i2 = tf.transpose(i1, [0,2,1,3,4])
    i3 = tf.reshape(i2, [1,1,8,4])
    return i3

def nm2_hd2(i0):
    # i0 = nd2_hm2(tf.reshape([i for i in range(2*4*4)], [2,1,4,4]))
    i1 = tf.reshape(i0, [1,1,2,4,4])
    i2 = tf.transpose(i1, [0,2,1,3,4])
    i3 = tf.reshape(i2, [2,1,4,4])
    return i3
