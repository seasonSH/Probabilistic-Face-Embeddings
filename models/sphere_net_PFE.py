from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
}

batch_norm_params_last = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': False,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}


def parametric_relu(x):
    num_channels = x.shape[-1].value
    with tf.variable_scope('p_re_lu'):
        alpha = tf.get_variable('alpha', (1,1,num_channels),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
        return tf.nn.relu(x) + alpha * tf.minimum(0.0, x)

def se_module(input_net, ratio=16, reuse = None, scope = None):
    with tf.variable_scope(scope, 'SE', [input_net], reuse=reuse):
        h,w,c = tuple([dim.value for dim in input_net.shape[1:4]])
        assert c % ratio == 0
        hidden_units = int(c / ratio)
        squeeze = slim.avg_pool2d(input_net, [h,w], padding='VALID')
        excitation = slim.flatten(squeeze)
        excitation = slim.fully_connected(excitation, hidden_units, scope='se_fc1',
                                weights_regularizer=None,
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.relu)
        excitation = slim.fully_connected(excitation, c, scope='se_fc2',
                                weights_regularizer=None,
                                weights_initializer=slim.xavier_initializer(), 
                                activation_fn=tf.nn.sigmoid)        
        excitation = tf.reshape(excitation, [-1,1,1,c])
        output_net = input_net * excitation

        return output_net

def conv_module(net, num_res_layers, num_kernels, trans_kernel_size=3, trans_stride=2,
                     use_se=False, reuse=None, scope=None):
    with tf.variable_scope(scope, 'conv', [net], reuse=reuse):
        net = slim.conv2d(net, num_kernels, kernel_size=trans_kernel_size, stride=trans_stride, padding='SAME',
                weights_initializer=slim.xavier_initializer()) 
        shortcut = net
        for i in range(num_res_layers):
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            net = slim.conv2d(net, num_kernels, kernel_size=3, stride=1, padding='SAME',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                biases_initializer=None)
            print('| ---- block_%d' % i)
            if use_se:
                net = se_module(net)
            net = net + shortcut
            shortcut = net
    return net

def inference(images, embedding_size=512, reuse=None, scope='SphereNet'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.0),
                        normalizer_fn=None, 
                        normalizer_params=None, 
                        activation_fn=parametric_relu):
        with tf.variable_scope('SphereNet', [images], reuse=reuse):
            # Fix the moving mean and std when training PFE 
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False): 

                print('SphereNet input shape:', [dim.value for dim in images.shape])
                
                model_version = '64' 
                num_layers, num_kernels = model_params[model_version]


                net = conv_module(images, num_layers[0], num_kernels[0], scope='conv1')
                print('module_1 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, num_layers[1], num_kernels[1], scope='conv2')
                print('module_2 shape:', [dim.value for dim in net.shape])
                
                net = conv_module(net, num_layers[2], num_kernels[2], scope='conv3')
                print('module_3 shape:', [dim.value for dim in net.shape])

                net = conv_module(net, num_layers[3], num_kernels[3], scope='conv4')
                print('module_4 shape:', [dim.value for dim in net.shape])

                net_ = net
                net = slim.flatten(net)

                mu = slim.fully_connected(net, embedding_size, scope='Bottleneck',
                                        weights_initializer=slim.xavier_initializer(),
                                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_last,
                                        activation_fn=None)
                
                # Output used for PFE
                mu = tf.nn.l2_normalize(mu, axis=1)
                conv_final = net
            
    return mu, conv_final
