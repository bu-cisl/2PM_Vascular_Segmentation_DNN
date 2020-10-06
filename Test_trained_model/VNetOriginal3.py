import tensorflow as tf
from Layers import convolution_3d, deconvolution_3d


def _convolution_block(layer_input, n_channels, num_convolutions, is_train):
    x = layer_input
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            x = tf.nn.relu(convolution_3d(x, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1]))
            # # batch normalization
            # mn, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
            # beta = tf.Variable(tf.zeros([n_channels]), name='beta')
            # gamma = tf.Variable(tf.ones([n_channels]), name='gamma')
            # x = tf.nn.batch_normalization(x, mn, var, beta, gamma, 1e-3)
            x = tf.layers.batch_normalization(x, training=is_train)
    return x + layer_input


def _convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions, is_train):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)

    with tf.variable_scope('conv_' + str(1)):
        x = convolution_3d(x, [3, 3, 3, n_channels * 2, n_channels], [1, 1, 1, 1, 1])

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            x = tf.nn.relu(convolution_3d(x, [3, 3, 3, n_channels, n_channels], [1, 1, 1, 1, 1]))
            # # batch normalization
            # mn, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
            # beta = tf.Variable(tf.zeros([n_channels]), name='beta')
            # gamma = tf.Variable(tf.ones([n_channels]), name='gamma')
            # x = tf.nn.batch_normalization(x, mn, var, beta, gamma, 1e-3)
            x = tf.layers.batch_normalization(x, training=is_train)
    return x + layer_input


def _down_convolution(layer_input, in_channels):
    with tf.variable_scope('down_convolution'):
        return tf.nn.relu(convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1]))


def _up_convolution(layer_input, output_shape, in_channels):
    with tf.variable_scope('up_convolution'):
        return tf.nn.relu(deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels],
                                           output_shape, [1, 2, 2, 2, 1]))


def v_net(tf_input, input_channels, output_channels=1, is_train=True, n_channels=4, name=None):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:

        tf_input: a rank 5 tensor with shape [batch_size, X, Y, Z, input_channels]
            where X, Y, Z are the spatial dimensions of the images and input_channels is
            the number of channels the images have;

        input_channels: the number of channels of the input images;

        output_channels: the number of desired output channels. v_net() will return
            a tensor with the same shape as tf_input but with a different number of channels
            i.e. [batch_size, x, y, z, output_channels].

        is_train: boolean flag for whether the network is being used for training or tesing.
            this is required for batch normalization.

        n_channels: the number of channels used internally in the network. In the original
            paper this number was 16. This number doubles at every level of the contracting path.
            See the image for better understanding of this number.

        name: name or scope of this network

    Returns:

        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
        equal to the logits of classifying the digit into one of 10 classes (the
        digits 0-9). keep_prob is a scalar placeholder for the probability of
        dropout.
    """
    with tf.variable_scope(name, 'VNet_3x3x3'):
        with tf.variable_scope('contracting_path'):
            # if the input has more than 1 channel it has to be expanded
            # because broadcasting only works for 1 input channel
            if input_channels == 1:
                c0 = tf.tile(tf_input, [1, 1, 1, 1, n_channels])
            else:
                with tf.variable_scope('level_0'):
                    c0 = tf.nn.relu(convolution_3d(tf_input, [3, 3, 3, input_channels, n_channels], [1, 1, 1, 1, 1]))

            with tf.variable_scope('level_1'):
                c1 = _convolution_block(c0, n_channels, 1, is_train)
                c12 = _down_convolution(c1, n_channels)

            with tf.variable_scope('level_2'):
                c2 = _convolution_block(c12, n_channels * 2, 2, is_train)
                c22 = _down_convolution(c2, n_channels * 2)

            with tf.variable_scope('level_3'):
                c3 = _convolution_block(c22, n_channels * 4, 3, is_train)
                c32 = _down_convolution(c3, n_channels * 4)

            with tf.variable_scope('level_4'):
                c4 = _convolution_block(c32, n_channels * 8, 3, is_train)
                c42 = _up_convolution(c4, tf.shape(c3), n_channels * 8)

            # with tf.variable_scope('level_5'):
            #     c5 = _convolution_block(c42, n_channels * 16, 3)
            #     c52 = _up_convolution(c5, tf.shape(c4), n_channels * 16)

        with tf.variable_scope('expanding_path'):

            # with tf.variable_scope('level_4'):
            #     e4 = _convolution_block_2(c52, c4, n_channels * 8, 3)
            #     e42 = _up_convolution(e4, tf.shape(c3), n_channels * 8)

            with tf.variable_scope('level_3'):
                e3 = _convolution_block_2(c42, c3, n_channels * 4, 3, is_train)
                e32 = _up_convolution(e3, tf.shape(c2), n_channels * 4)

            with tf.variable_scope('level_2'):
                e2 = _convolution_block_2(e32, c2, n_channels * 2, 2, is_train)
                e22 = _up_convolution(e2, tf.shape(c1), n_channels * 2)

            with tf.variable_scope('level_1'):
                e1 = _convolution_block_2(e22, c1, n_channels, 1, is_train)
                with tf.variable_scope('output_layer'):
                    logits = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

        return logits
