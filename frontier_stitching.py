import tensorflow as tf


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
