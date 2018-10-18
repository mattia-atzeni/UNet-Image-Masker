from keras.losses import binary_crossentropy
import keras.backend as K
import cv2
import numpy as np
import tensorflow as tf


def dice_loss(y_true, y_pred):
    #smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))


def gauss2D(shape=(3,3),sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


log = np.array([
    [0.0448, 0.0468, 0.0564, 0.0468, 0.0448],
    [0.0468, 0.3167, 0.7146, 0.3167, 0.0468],
    [0.0564, 0.7146, -4.9048, 0.7146, 0.0564],
    [0.0468, 0.3167, 0.7146, 0.3167, 0.0468],
    [0.0448, 0.0468, 0.0564, 0.0468, 0.0448]]).astype(np.float32)


def weights_mask(mask):
    log_kernel = tf.convert_to_tensor(log)
    log_kernel = tf.reshape(log_kernel, [5, 5, 1, 1])
    log_kernel = tf.to_float(log_kernel)
    mask = tf.to_float(mask)
    edges = tf.nn.conv2d(mask, log_kernel, padding='SAME', strides=[1, 1, 1, 1])
    edges = edges > 0.95
    edges = tf.to_float(edges)
    gauss_kernel = tf.convert_to_tensor(gauss2D((5, 5), 2))
    gauss_kernel = tf.reshape(gauss_kernel, [5, 5, 1, 1])
    gauss_kernel = tf.to_float(gauss_kernel)
    return tf.nn.conv2d(edges, gauss_kernel, padding='SAME', strides=[1, 1, 1, 1])


def edges_dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.flatten(weights_mask(y_true))
    intersection = K.sum(y_true_f * y_pred_f)
    eq = tf.equal(y_true_f, y_pred_f)
    eq = tf.to_float(eq)
    weightedEdges = K.sum(eq * weights)
    return (2. * intersection + weightedEdges) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.sum(weights))


def bce_edges(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - edges_dice_loss(y_true, y_pred))


# weight: weighted tensor(same shape with mask image)
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
                                          (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
           weighted_dice_loss(y_true, y_pred, weight)
    return loss
