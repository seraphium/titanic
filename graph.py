import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pandas as pd


tf.logging.set_verbosity(tf.logging.INFO)


def convert_data_to_tensors(x, y=None):
    inputs = tf.constant(x)
    inputs.set_shape([None, 8])
    outputs = None
    if y is not None:
        outputs = tf.constant(y)
        outputs.set_shape([None, 2])
    return inputs, outputs


def get_network(inputs, is_training, scope="network"):
    with tf.variable_scope(scope, "network", [inputs]):
        endpoints = {}
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.sigmoid,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            net = slim.fully_connected(inputs, 10, scope="fc1")
            endpoints['fc1']=net
            #net = slim.dropout(net, 0.8, scope='dropout1',  is_training=is_training)
            net = slim.fully_connected(net, 10, scope="fc2")
            endpoints['fc2']=net
            net = slim.fully_connected(net, 10, scope="fc3")
            endpoints['fc3']=net
            #net = slim.dropout(net, 0.8, scope='dropout2', is_training=is_training)
            net = slim.fully_connected(net, 2, scope="output")
            output = tf.nn.softmax(net, name="predictions")
            endpoints["prediction"] = output
            return output, endpoints

