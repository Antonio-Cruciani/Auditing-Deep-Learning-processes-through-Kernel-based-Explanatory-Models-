import tensorflow as tf
import sys


def activate(input_value, activation):
    """
    It corresponds to the activation function of a neuron

    :param input_value: it is the neuron input
    :param activation: the type of the activation function. It must be one of the following strings:
        tanh, sigmoid, relu, softplus, elu
    :return: the output of the neuron, after the activation function is applied
    """
    if activation == 'tanh':
        act_value = tf.nn.tanh(input_value, name="tanh")
    elif activation == 'sigmoid':
        act_value = tf.nn.sigmoid(input_value, name="sigmoid")
    elif activation == 'relu':
        act_value = tf.nn.relu(input_value, name="relu")
    elif activation == 'softplus':
        act_value = tf.nn.softplus(input_value, name="softplus")
    elif activation == 'elu':
        act_value = tf.nn.elu(input_value, name="elu")
    elif activation == 'lin':
        act_value = input_value
    else:
        sys.exit('Unrecognized activation value {}'.format(activation))

    return act_value


def softmax_cross_entropy(scores, gold):
    """ Computes the empirical error, i.e., the classification loss in the mini-batch

    :param scores: the predicted values
    :param gold: the gold values (they are one-hot vectors)
    :return: the classification loss
    """
    logsoftmax = tf.log(tf.nn.softmax(scores) + 1e-9)
    return tf.negative(tf.reduce_sum(tf.multiply(logsoftmax, gold), 1))