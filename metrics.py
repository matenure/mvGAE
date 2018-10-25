import tensorflow as tf
from sklearn.metrics import accuracy_score


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # accuracy_all = tf.cast(correct_prediction, tf.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    # mask /= tf.reduce_mean(mask)
    # accuracy_all *= mask
    # return tf.reduce_mean(accuracy_all)

    correct_prediction = tf.equal(labels>=0.5,preds>=0.5)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_bilinearsigmoid_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    maskIndex = tf.where(mask>0)
    maskIndex = tf.cast(maskIndex, dtype=tf.int32)
    maskedPreds = tf.gather_nd(preds, maskIndex)
    maskedLabels = tf.gather_nd(labels, maskIndex)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=maskedPreds, targets=maskedLabels)
    return tf.reduce_mean(loss)



def masked_squreloss(preds, labels, mask):
    # maskIndex = tf.where(mask>0)
    # maskIndex = tf.cast(maskIndex, dtype=tf.int32)
    # maskedPreds = tf.gather_nd(preds, maskIndex)
    # maskedLabels = tf.gather_nd(labels, maskIndex)
    # loss = tf.squared_difference(maskedPreds,maskedLabels)
    # return tf.reduce_mean(loss)
    loss = tf.squared_difference(preds,labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)



def masked_bilinearsoftmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    maskIndex = tf.where(mask>0)
    maskIndex = tf.cast(maskIndex, dtype=tf.int32)
    maskedPreds = tf.gather_nd(preds, maskIndex)
    maskedLabels = tf.gather_nd(labels, maskIndex)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=maskedPreds, labels=maskedLabels)
    return tf.reduce_mean(loss)

def masked_bilinear_accuray(preds, labels, mask):
    maskIndex = tf.where(mask > 0)
    maskIndex = tf.cast(maskIndex, dtype=tf.int32)
    maskedPreds = tf.gather_nd(preds, maskIndex)
    maskedLabels = tf.gather_nd(labels, maskIndex)
    correct_prediction = tf.equal(maskedLabels >= 0.5, maskedPreds >= 0.5)
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)