import numpy as np
import tensorflow as tf
import after_test.utils as utils
slim = tf.contrib.slim


def vnect(x, j):
    with tf.compat.v1.variable_scope('vnect') as vs:
        with tf.compat.v1.variable_scope('block1_a'):
            init_x = x
            x = slim.conv2d(x, 512, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 512, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 1024, 1, 1, activation_fn=None, data_format='NCHW', normalizer_fn=slim.batch_norm)
            init_x = slim.conv2d(init_x, 1024, 1, 1, activation_fn=None, data_format='NCHW')
            x = x + init_x
        with tf.compat.v1.variable_scope('block1_b'):
            x = slim.conv2d(x, 256, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.separable_conv2d(x, 128, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 256, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
        with tf.compat.v1.variable_scope('upsample_conv'):
            x = utils.upscale(x, 2, 'NCHW')
            non_j_x = slim.conv2d(x, 128, 4, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            j_x = slim.conv2d(x, 3 * j, 4, 1, activation_fn=None, data_format='NCHW', normalizer_fn=slim.batch_norm)
            dx, dy, dz = tf.split(j_x, 3, axis=1)
            d_j = tf.sqrt(tf.maximum(dx*dx+dy*dy+dz*dz, 0.))
            # d_j = tf.concat([dx, dy, dz], axis=1)  # temptemptemptemptemp
            x = tf.concat([non_j_x, d_j, j_x], axis=1)
        with tf.compat.v1.variable_scope('lsat'):
            x = slim.conv2d(x, 128, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 4*j, 1, 1, activation_fn=None, data_format='NCHW')
    var = tf.contrib.framework.get_variables(vs)
    return x, var  # [256^2 * 4*j]


def movnect(x, j):
    with tf.compat.v1.variable_scope('movnect') as vs:
        with tf.compat.v1.variable_scope('block13_a'):
            init_x = x
            x = slim.conv2d(x, 368, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.separable_conv2d(x, 368, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 256, 1, 1, activation_fn=None, data_format='NCHW', normalizer_fn=slim.batch_norm)
            init_x = slim.conv2d(init_x, 256, 1, 1, activation_fn=None, data_format='NCHW')
            x = x + init_x
        with tf.compat.v1.variable_scope('block13_b'):
            x = slim.conv2d(x, 192, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.separable_conv2d(x, 192, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.conv2d(x, 192, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
        with tf.compat.v1.variable_scope('upsample_conv'):
            x = utils.upscale(x, 2, data_format='NCHW')
            non_j_x = slim.conv2d(x, 128, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            j_x = slim.conv2d(x, 3 * j, 1, 1, activation_fn=None, data_format='NCHW', normalizer_fn=slim.batch_norm)
            dx, dy, dz = tf.split(j_x, 3, axis=1)
            d_j = tf.abs(dx) + tf.abs(dy) + tf.abs(dz)
            x = tf.concat([non_j_x, d_j, j_x], axis=1)
        with tf.compat.v1.variable_scope('lsat'):
            x = slim.conv2d(x, 128, 1, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.separable_conv2d(x, 128, 3, 1, activation_fn=tf.nn.relu, data_format='NCHW', normalizer_fn=slim.batch_norm)
            x = slim.separable_conv2d(x, 4*j, 1, 1, activation_fn=None, data_format='NCHW')
    var = tf.contrib.framework.get_variables(vs)
    return x, var  # [256^2 * 4*j]


def mobile_net(x):
    x = slim.separable_conv2d(x, 32, 3, 2, activation_fn=tf.nn.relu6)  # 112^2 * 32
    x = bottleneck(x, 1, 16, 1, 1, i=0)
    x = bottleneck(x, 6, 24, 2, 2, i=1)
    x = bottleneck(x, 6, 32, 3, 2, i=2)
    x = bottleneck(x, 6, 64, 4, 2, i=3)
    x = bottleneck(x, 6, 96, 3, 1, i=4)  # bottleneck 0 ~ 12
    return x  # 14^2 * 96


def bottleneck(x, t, c, n, s, i=0):
    for j in range(n):
        with tf.compat.v1.variable_scope('bottleneck_{}'.format(i*1000 + j)):
            init_x = x
            x = slim.conv2d(x, x.shape[1] * t, c, 1, activation_fn=tf.nn.relu6)
            if j is 0:
                x = slim.separable_conv2d(x, x.shape[1], c, s, activation_fn=tf.nn.relu6)
            else:
                x = slim.separable_conv2d(x, x.shape[1], c, 1, activation_fn=tf.nn.relu6)
            x = slim.conv2d(x, int(x.shape[1]/t), 1, activation_fn=None)
            if j is not 0:
                x = x + init_x
    return x
