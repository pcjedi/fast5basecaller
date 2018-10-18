import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell

def conv_layer(indata, out_channels, batch_norm=True, k=1, padding='SAME', stride=1):
    strides = [1, stride, stride, 1]
    in_channels = indata.get_shape()[-1]
    shape = [1, k, in_channels, out_channels]
    W = tf.get_variable('weights', shape=shape)
    conv_out = tf.nn.conv2d(indata, W, strides=strides, padding=padding)
    if batch_norm:
        with tf.variable_scope('batch_normalization'):
            mean, variance = tf.nn.moments(conv_out, [0, 1, 2])
            scale = tf.get_variable('scale', shape=[out_channels])
            offset = tf.get_variable('offset', shape=[out_channels])
            conv_out = tf.nn.batch_normalization(conv_out, 
                                                 mean=mean, 
                                                 variance=variance, 
                                                 scale=scale, 
                                                 offset=offset,
                                                 variance_epsilon=1e-5)
    return conv_out

def residual_layer(indata, out_channels, batch_norm=True, stride=1):
    in_channel = indata.get_shape()[-1]
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(indata, out_channels, batch_norm)
    
    with tf.variable_scope('conv2'):
        with tf.variable_scope('a'):
            conv2 = conv_layer(indata, out_channels)
            conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('b'):
            conv2 = conv_layer(conv2, out_channels, k=3)
            conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('c'):
            conv2 = conv_layer(conv2, out_channels)
            conv2 = tf.nn.relu(conv2)
    
    return tf.nn.relu(conv1 + conv2)

batch_size = 200
segment_length = 400
num_features = 1
neurons_per_layer = 100
layers = 3
class_n = 5

x = tf.placeholder(tf.float32, shape=[batch_size, segment_length, num_features], name = 'x')
sequence_length = tf.placeholder(tf.int32, shape=[batch_size], name = 'sequence_length')

net = tf.reshape(x, [batch_size, 1, segment_length, num_features])
out_channel = 256

with tf.name_scope('CNN'):
    for i in range(3):
        with tf.variable_scope('residual_layer_'+str(i)):
            net = residual_layer(net, out_channel)
        
with tf.name_scope('RNN'):
    conv_all = tf.reshape(net, [batch_size, segment_length, out_channel])
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = MultiRNNCell([LSTMCell(neurons_per_layer) for _ in range(layers)]), 
        cell_bw = MultiRNNCell([LSTMCell(neurons_per_layer) for _ in range(layers)]), 
        inputs = conv_all, 
        sequence_length = sequence_length,
        dtype = tf.float32
    )

with tf.name_scope('collapse'):

    outputs = tf.concat(outputs, 2)
    outputs = tf.reshape(outputs, [batch_size, segment_length, 2, neurons_per_layer], name = 'outputs')

    W_collapse = tf.get_variable('W_collapse', shape = [2, neurons_per_layer])
    b_collapse = tf.get_variable('b_callapse', shape = [neurons_per_layer])

    collapsed = tf.multiply(outputs, W_collapse)
    collapsed = tf.reduce_sum(collapsed, axis=2)
    collapsed = tf.nn.bias_add(collapsed, b_collapse)
    collapsed = tf.reshape(collapsed, [batch_size*segment_length,neurons_per_layer], name = 'collapsed')

with tf.name_scope('FF'):
    
    W_last = tf.get_variable('W_last', shape = [neurons_per_layer, class_n])
    b_last = tf.get_variable('b_last', shape = [class_n])

    logits = tf.matmul(collapsed, W_last)
    logits = tf.add(logits, b_last)
    logits = tf.reshape(logits, [batch_size, segment_length, class_n], name="logits")
