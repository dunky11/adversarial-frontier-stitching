import tensorflow as tf
from helpers import binomial


def fast_gradient_signed(x, y, model, eps):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        loss = model.loss(y, y_pred)
    gradient = tape.gradient(loss, x)
    sign = tf.sign(gradient)
    return x + eps * sign


def gen_adversaries(model, l, dataset, eps):
    true_advs = []
    false_advs = []
    max_true_advs = max_false_advs = l // 2
    for x, y in dataset:
        # generate adversaries
        x_advs = fast_gradient_signed(x, y, model, eps)

        y_preds = tf.argmax(model(x), axis=1)
        y_pred_advs = tf.argmax(model(x_advs), axis=1)
        for x_adv, y_pred_adv, y_pred, y_true in zip(x_advs, y_pred_advs, y_preds, y):
            # x_adv is a true adversary
            if y_pred == y_true and y_pred_adv != y_true and len(true_advs) < max_true_advs:
                true_advs.append((x_adv, y_true))

            # x_adv is a false adversary
            if y_pred == y_true and y_pred_adv == y_true and len(false_advs) < max_false_advs:
                false_advs.append((x_adv, y_true))

            if len(true_advs) == max_true_advs and len(false_advs) == max_false_advs:
                return true_advs, false_advs

    return true_advs, false_advs


# finds a value for theta (maximum number of errors tolerated for verification)
def find_tolerance(key_length, threshold):
    theta = 0
    factor = 2 ** (-key_length)
    while(True):
        s = 0
        for z in range(theta + 1):
            s += binomial(key_length, z)
        if factor * s >= threshold:
            return theta
        theta += 1


def verify(model, key_set, threshold=0.05):
    m_k = 0
    length = 0
    for x, y in key_set:
        length += len(x)
        preds = tf.argmax(model(x), axis=1)
        m_k += tf.reduce_sum(tf.cast(preds != y, tf.int32))
    theta = find_tolerance(length, threshold)
    m_k = m_k.numpy()
    return {
        "success": m_k < theta,
        "false_preds": m_k,
        "max_fals_pred_tolerance": theta
    }
