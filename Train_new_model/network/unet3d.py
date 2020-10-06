import tensorflow as tf
from Layers import convolution_3d, deconvolution_3d, prelu


def _analysis_convolution_block(input_, in_channels, out_channels):
    feature_map = input_
    with tf.variable_scope('conv_1'):
        feature_map = convolution_3d(feature_map,
                                     [3, 3, 3, in_channels, out_channels // 2],
                                     [1, 1, 1, 1, 1])
        feature_map = prelu(feature_map)
    with tf.variable_scope('conv_2'):
        feature_map = convolution_3d(feature_map,
                                     [3, 3, 3, out_channels // 2, out_channels],
                                     [1, 1, 1, 1, 1])
        feature_map = prelu(feature_map)
    return feature_map


def _synthesis_convolution_block(input_, in_channels, out_channels, feature_from_analysis):
    feature_map = tf.concat((input_, feature_from_analysis), axis=-1)

    with tf.variable_scope('conv_1'):
        feature_map = convolution_3d(feature_map,
                                     [3, 3, 3, in_channels + in_channels // 2, in_channels // 2],
                                     [1, 1, 1, 1, 1])
        feature_map = prelu(feature_map)

    with tf.variable_scope('conv_2'):
        feature_map = convolution_3d(feature_map,
                                     [3, 3, 3, in_channels // 2, out_channels],
                                     [1, 1, 1, 1, 1])
        feature_map = prelu(feature_map)
    return feature_map


def _down_convolution(input_, in_channels):
    with tf.variable_scope('down_convolution'):
        return convolution_3d(input_, [2, 2, 2, in_channels, in_channels], [1, 2, 2, 2, 1])


def _up_convolution(input_, output_shape, in_channels):
    """

    :param input_: data shape [batch, depth, height, width, in_channels]
    :param output_shape: must match input dimensions [batch, depth, height, width, in_channels]
    :param in_channels:
    :return:
    """
    with tf.variable_scope('up_convolution'):
        return deconvolution_3d(input_, [2, 2, 2, in_channels, in_channels], output_shape, [1, 2, 2, 2, 1])


def u_net(input_, input_channels, output_channels=None):
    # original_shape = tf.shape(input_)
    if output_channels is None:
        output_channels = input_channels
    num_layers = 3
    channels = 16
    analysis_feature_maps = list(range(num_layers + 1))
    synthesis_feature_maps = list(range(num_layers + 1))

    with tf.variable_scope('analysis_path'):
        with tf.variable_scope('level_0'):
            analysis_feature_maps[0] = _analysis_convolution_block(
                input_, input_channels, channels)
        for i in range(1, num_layers + 1):
            # default i = 1, 2, 3
            with tf.variable_scope('level_' + str(i)):
                analysis_feature_maps[i] = _down_convolution(
                    analysis_feature_maps[i - 1], channels)
                analysis_feature_maps[i] = _analysis_convolution_block(
                    analysis_feature_maps[i], channels, channels * 2)
                channels *= 2

    synthesis_feature_maps[num_layers] = analysis_feature_maps[num_layers]
    with tf.variable_scope('synthesis_path'):
        for i in range(num_layers - 1, -1, -1):
            # default i = 2, 1, 0
            with tf.variable_scope('level_' + str(i)):
                output_shape = tf.shape(analysis_feature_maps[i])*[1, 1, 1, 1, 2]
                # the size should match but channels are not the same?
                # *[1, 1, 1, 1, 2] is to avoid dimension problems. Is that good?
                synthesis_feature_maps[i] = _up_convolution(
                    synthesis_feature_maps[i + 1], output_shape, channels
                )
                synthesis_feature_maps[i] = _synthesis_convolution_block(
                    synthesis_feature_maps[i], channels, channels // 2, analysis_feature_maps[i]
                )
                channels //= 2

    with tf.variable_scope('output_layer'):
        logits = convolution_3d(
            synthesis_feature_maps[0],
            [1, 1, 1, channels, output_channels],
            [1, 1, 1, 1, 1]
        )
    return logits
