#############################################################
# file: bal_cross_ent_fp_correct.py
# Implements balanced binary cross-entropy with false +ve
# correction as in the paper: DeepVesselNet (2018)
#
# Author: Waleed Tahir
# waleedt@bu.edu
# 23 July 2018
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, logits, labels, name=None, **kwargs):
        pass


# def bal_cross_ent(y_smx, y, shape_cube):
#     """
#     Implements balanced binary cross-entropy with false +ve
#     correction as in the paper: DeepVesselNet (2018)
#
#     :param y_smx: output data on which softmax has been applied, size of
#         y_smx, y_predicted & y is: [batch_size, nx, ny, nz, out_channels]
#         where out_channels=2
#     :param y: true labels
#     :param shape_cube: [batch_size, nx, ny, nz]
#     :return:
#     """
#     n_p_n = tf.reduce_sum(y, axis=(0, 1, 2, 3))  # [fp_rate, fn_rate] = [|Yf+|,|Yf-|]
#     n_p_n_norm = tf.divide(n_p_n, shape_cube[0] * shape_cube[1] * shape_cube[2] * shape_cube[3])
#     n_n_norm_cube = tf.scalar_mul(n_p_n_norm[1], tf.ones(shape_cube, tf.float32))
#     n_p_norm_cube = tf.scalar_mul(n_p_n_norm[0], tf.ones(shape_cube, tf.float32))
#     n_n_p_norm_cube = tf.stack([n_n_norm_cube, n_p_norm_cube], axis=-1)
#     # compute balanced cross-entropy L1
#     bal_cross_entropy_cube = tf.reduce_sum(tf.multiply(n_n_p_norm_cube, tf.multiply(y, tf.log(y_smx))), axis=-1)
#     bal_cross_entropy = -tf.reduce_sum(bal_cross_entropy_cube, name='bal_cross_entropy')
#     return bal_cross_entropy
#
#
# def bal_cross_ent_fp_correct(y_smx, y_predicted, y, shape_cube):
#     """
#     Implements balanced binary cross-entropy with false +ve
#     correction as in the paper: DeepVesselNet (2018)
#
#     :param y_smx: output data on which softmax has been applied, size of
#         y_smx, y_predicted & y is: [batch_size, nx, ny, nz, out_channels]
#         where out_channels=2
#     :param y_predicted: predicted labels
#     :param y: true labels
#     :param shape_cube: [batch_size, nx, ny, nz]
#     :return:
#     """
#     n_p_n = tf.reduce_sum(y, axis=(0, 1, 2, 3))  # [fp_rate, fn_rate] = [|Yf+|,|Yf-|]                # INCLUDE
#     n_p_n_recip = tf.reciprocal(n_p_n)
#     n_p_recip_cube = tf.scalar_mul(n_p_n_recip[0], tf.ones(shape_cube, tf.float32))
#     n_n_recip_cube = tf.scalar_mul(n_p_n_recip[1], tf.ones(shape_cube, tf.float32))
#     n_p_n_recip_cube = tf.stack([n_p_recip_cube, n_n_recip_cube], axis=-1)
#     # compute balanced cross-entropy L1
#     cross_entropy_l1_cube = tf.reduce_sum(tf.multiply(n_p_n_recip_cube, tf.multiply(y, tf.log(y_smx))), axis=-1)
#     cross_entropy_l1 = -tf.reduce_sum(cross_entropy_l1_cube, name='cross_entropy_l1')
#
#     # L2
#     # indexes of false positives and negatives
#     y_fp_fn = tf.multiply(tf.reverse(y, [-1]), y_predicted)
#     # 1/|Yf+| and 1/|Yf-|
#     n_fp_fn = tf.reduce_sum(y_fp_fn, axis=(0, 1, 2, 3))  # [fp_rate, fn_rate] = [|Yf+|,|Yf-|]
#     n_fp_fn_recip = tf.reciprocal(n_fp_fn)
#     # gamma1 and gamma2
#     halves_cube = tf.scalar_mul(0.5, tf.ones(shape_cube, tf.float32))
#     gamma1 = 0.5 + tf.scalar_mul(n_fp_fn_recip[0], tf.reduce_sum(
#         tf.multiply(y_fp_fn[:, :, :, :, 0], tf.abs(tf.subtract(y_smx[:, :, :, :, 1], halves_cube)))))
#     gamma2 = 0.5 + tf.scalar_mul(n_fp_fn_recip[1], tf.reduce_sum(
#         tf.multiply(y_fp_fn[:, :, :, :, 1], tf.abs(tf.subtract(y_smx[:, :, :, :, 0], halves_cube)))))
#     # gamma1/|Yf+| and gamma2/|Yf-|
#
#     g1_fp_recip_cube = tf.scalar_mul(tf.scalar_mul(n_fp_fn_recip[0], gamma1), tf.ones(shape_cube, tf.float32))
#     g2_fn_recip_cube = tf.scalar_mul(tf.scalar_mul(n_fp_fn_recip[1], gamma2), tf.ones(shape_cube, tf.float32))
#     g_fp_fn_recip_cube = tf.stack([g1_fp_recip_cube, g2_fn_recip_cube], axis=-1)
#     # compute balanced cross-entropy L2
#     cross_entropy_l2_cube = tf.reduce_sum(
#         tf.multiply(g_fp_fn_recip_cube, tf.multiply(y_fp_fn, tf.log(tf.reverse(y_smx, [-1])))), axis=-1)
#     cross_entropy_l2 = -tf.reduce_mean(cross_entropy_l2_cube, name='cross_entropy_l2')
#
#     # L = L1 + L2
#     bal_cross_entropy_fp = cross_entropy_l1 + cross_entropy_l2
#
#     return bal_cross_entropy_fp


def sigmoid_cross_entropy_balanced(logits, labels, name: str = None, **kwargs):
    """
    Implements balanced binary cross-entropy with false +ve
    correction as in the paper: DeepVesselNet (2018)

    Balanced cross entropy is:
        targets * -log(sigmoid(logits)) * pos_weight
        + (1 - targets) * -log(1 - sigmoid(logits))

    Since pos_weight > 1, fp increase and fn decrease, recall increase and precision decrease

    :param name: tensor name
    :param logits: output data on which softmax has NOT been applied, size of
        logits & labels are: [batch_size, nx, ny, nz, out_channels]
        where out_channels=1
    :param labels: true labels
    :param kwargs:
        pos_weight: Fixed positive weight if set. If it is None, will use
        negative/positive as positive weight
    :return:
    """

    try:
        pos_weight = kwargs['pos_weight']
    except KeyError:
        pos_weight = None
    # compute balanced cross-entropy L1
    with tf.name_scope(name, 'balanced_cross_entropy', (logits, labels)) as scope:
        labels = tf.stop_gradient(labels, "label_gradient_stopped")
        if pos_weight is None:  # original function
            pos_rate = tf.reduce_mean(labels)
            pos_weight = (1 - pos_rate) / (pos_rate + 1e-3)
            # use 0.001 to avoid divided by zero
            # p_w = negative rate / Positive rate
        else:  # fix weight
            pos_rate = 1 / (pos_weight + 1)  # "Make" a pos_rate to normalize result

        bal_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
            targets=labels,
            logits=logits,
            pos_weight=pos_weight) * pos_rate
        # Actually p_w = Negative rate, n_w = Positive rate
        return tf.reduce_mean(bal_cross_entropy, name=scope)


def sigmoid_dice(logits, labels, name=None):
    """
    Implements dice loss as in the paper: V-Net

    :param name: tensor name
    :param logits: output data on which softmax has NOT been applied, size of
        logits & labels are: [batch_size, nx, ny, nz, out_channels]
        where out_channels=1
    :param labels: true labels
    :return:
    """
    with tf.name_scope(name, 'dice'):
        labels = tf.stop_gradient(labels, "label_gradient_stopped")
        predict = tf.sigmoid(logits)
        dice = (2 * tf.reduce_sum(predict * labels, name='sum_pi_gi')
                / (tf.reduce_sum(predict ** 2, name='sum_pi_2')
                   + tf.reduce_sum(labels, name='sum_gi_2'))
                )
        return 1 - dice


def sigmoid_cross_entropy(logits, labels, name=None):
    with tf.name_scope(name, "cross_entropy") as scope:
        labels = tf.stop_gradient(labels, "label_gradient_stopped")
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        )
        return tf.reduce_mean(cross_entropy, name=scope)


def derivative_mean(img, labels, name=None):
    with tf.name_scope(name, "sobel_sum"):
        sobel_one_direction = np.array([[[1, 2, 1],
                                         [2, 4, 2],
                                         [1, 2, 1]],

                                        [[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]],

                                        [[-1, -2, -1],
                                         [-2, -4, -2],
                                         [-1, -2, -1]]], dtype=np.float32)
        sobel = np.stack([sobel_one_direction,
                          sobel_one_direction.transpose((2, 0, 1)),
                          sobel_one_direction.transpose((1, 2, 0))],
                         axis=-1)

        sobel = tf.constant(sobel[..., np.newaxis, :])
        derivative = tf.reduce_mean(tf.abs(tf.nn.conv3d(tf.sigmoid(img), sobel, (1,) * 5, "VALID"))) # tf.nn.conv3d filter shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
        sup = sigmoid_dice(img, labels)
        return 0.2 * derivative + 0.8 * sup

def total_variation_balanced_cross_entropy(img, labels, name: str = None, **kwargs):

    try:
        alpha = kwargs['alpha']
    except KeyError:
        alpha = 1

    with tf.name_scope(name, "tv_bce"):
        print("*********** alpha:%1.10f ************"%alpha)
        sobel_one_direction = np.array([[[1, 2, 1],
                                         [2, 4, 2],
                                         [1, 2, 1]],

                                        [[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]],

                                        [[-1, -2, -1],
                                         [-2, -4, -2],
                                         [-1, -2, -1]]], dtype=np.float32)
        sobel = np.stack([sobel_one_direction,
                          sobel_one_direction.transpose((2, 0, 1)),
                          sobel_one_direction.transpose((1, 2, 0))],
                         axis=-1) #3x3x3x3

        sobel = tf.constant(sobel[..., np.newaxis, :]) #3x3x3x1x3
        tv_loss = tf.reduce_sum(tf.abs(tf.nn.conv3d(tf.sigmoid(img), sobel, (1,) * 5, "VALID"))) # tf.nn.conv3d filter shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
        bce_loss = sigmoid_cross_entropy_balanced(img, labels)
        total_loss = bce_loss + alpha * tv_loss
        return total_loss

# class SigmoidDice(LossFunction):
#     def __call__(self, logits, labels, name=None, **kwargs):
#         """
#             Implements dice loss as in the paper: V-Net
#
#             :param name: tensor name
#             :param logits: output data on which softmax has NOT been applied, size of
#                 logits & labels are: [batch_size, nx, ny, nz, out_channels]
#                 where out_channels=1
#             :param labels: true labels
#             :return:
#         """
#         with tf.name_scope(name, 'dice'):
#             labels = tf.stop_gradient(labels, "label_gradient_stopped")
#             predict = tf.sigmoid(logits)
#             dice = (2 * tf.reduce_sum(predict * labels, name='sum_pi_gi')
#                     / (tf.reduce_sum(predict ** 2, name='sum_pi_2')
#                        + tf.reduce_sum(labels, name='sum_gi_2'))
#                     )
#             return 1 - dice
