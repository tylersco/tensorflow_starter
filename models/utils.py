'''
Decorators adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

import functools
import numpy as np
import tensorflow as tf

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def doublewrap(function):
    '''
    A decorator around a decorator allowing use of the original decorator
    without parentheses if no arguments are provided. All arguments
    must be optional.
    '''
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator

@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    '''
    A decorator for functions that define Tensorflow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result, so that operations are only added to the graph once.
    '''
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def _stride(s):
    return [1, s, s, 1]

def _relu(x):
    return tf.nn.relu(x)

def _sigmoid(x):
    return tf.nn.sigmoid(x)

def _tanh(x):
    return tf.nn.tanh(x)

def _conv(name, x, filter_size, in_size, out_size, stride, padding='SAME', bias=True, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        n = filter_size * filter_size * out_size
        weights = tf.get_variable(
            'conv_weights', [filter_size, filter_size, in_size, out_size],
            tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n))
        )

        res = tf.conv2d(x, weights, _stride(stride), padding=padding)

        if bias:
            biases = tf.get_variable(
                'conv_biases', [out_size], tf.float32,
                initializer=tf.zeros_initializer()
            )
            res += biases

    return res

def _fully_connected(name, x, out_size, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        # NOTE: Factor used in initializer may need to be tuned
        weights = tf.get_variable(
            'fc_weights', [x.get_shape()[1], out_size],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0)
        )
        biases = tf.get_variable(
            'fc_biases', [out_size], initializer=tf.zeros_initializer()
        )
        return tf.nn.xw_plus_b(x, weights, biases)
